from dataclasses import dataclass
from typing import List, Optional

import tvm
from tvm.contrib.nvcc import parse_compute_version
from tvm import relax, te, tir, topi
from tvm.script import tir as T
from tvm.relax.expr_functor import visitor

from . import tir_utils
from .quantization import QuantizationSpec, QuantSpecUpdater
from .quantization import FQuantize, convert_TE_func
from .group_quantization import GroupQuantizationSpec


@dataclass
class FTRowwiseQuantizationSpec(QuantizationSpec):
    """The quantization specification for the FasterTransformer kernel."""

    def __init__(self, dtype, nbit):
        super().__init__(dtype)
        self.nbit = nbit

        if tvm.cuda(0).exist:
            major, minor = parse_compute_version(tvm.cuda(0).compute_version)
            if major == 8:
                self.sm = 80
            else:
                self.sm = 10 * major + minor
        else:
            self.sm = None

    def get_quantize_func(self, param_info: relax.TensorStructInfo) -> Optional[FQuantize]:
        assert self.sm is not None

        def f_quantize(bb: relax.BlockBuilder, inputs: List[relax.Expr]):
            encoded_data = bb.emit_te(
                encoding_func(
                    self.nbit,
                    8,
                    dtype=self.dtype,
                ),
                inputs[0],
                primfunc_name_hint="encode",
            )
            packed_weight = bb.normalize(encoded_data[0])
            encoded_weight = bb.emit(
                relax.call_pure_packed(
                    "cutlass.ft_preprocess_weight",
                    packed_weight,
                    self.sm,
                    self.nbit == 4,
                    sinfo_args=packed_weight.struct_info,
                )
            )
            return bb.emit(relax.Tuple([encoded_weight, encoded_data[1]]))

        return f_quantize

    def get_dequantize_func(
        self,
        param_info: relax.TensorStructInfo,
        qparam_info: List[relax.TensorStructInfo],
    ) -> Optional[FQuantize]:
        return convert_TE_func(
            decoding_func(
                self.nbit,
                storage_nbit=8,
            ),
            func_name="decode",
        )


def encoding_func(nbit: int, storage_nbit: int, dtype: str = "float32"):
    def te_encode_sym(weight: te.Tensor):
        n_float_per_int = storage_nbit // nbit
        max_int_value = (1 << (nbit - 1)) - 1

        scale_min_shape = (weight.shape[0],)
        k = te.reduce_axis((0, weight.shape[1]), name="k")
        max_abs_value = te.compute(
            shape=scale_min_shape,
            fcompute=lambda i: te.max(te.abs(weight[i, k]), axis=k),
            name="max_abs_value",
        )

        def f_compute_scale(i):
            max_value = tir.max(tir.Cast(dtype, max_abs_value[i]), tir.const(1e-4, dtype))
            return max_value / tir.const(max_int_value + 1, dtype)

        scale = te.compute(shape=scale_min_shape, fcompute=f_compute_scale, name="scale")
        storage_dtype = "int" + str(storage_nbit)

        def f_scale_weight(i, j):
            w_scaled = tir.round(tir.Cast(dtype, weight[i, j]) / scale[i])
            w_scaled = T.min(
                T.max(w_scaled, tir.const(-max_int_value - 1, dtype)),
                tir.const(max_int_value, dtype),
            ).astype(storage_dtype)
            if n_float_per_int == 1:
                return w_scaled
            return w_scaled & tir.const((1 << nbit) - 1, storage_dtype)

        n_i32 = tir.ceildiv(weight.shape[0], n_float_per_int)

        if n_float_per_int == 1:
            w_gathered = te.compute(
                shape=(weight.shape[1], n_i32),
                fcompute=lambda j, i: f_scale_weight(i, j),
                name="w_gathered",
            )
        else:
            k = te.reduce_axis((0, n_float_per_int), name="k")
            reducer = te.comm_reducer(
                fcombine=lambda x, y: tir.bitwise_or(x, y),
                fidentity=lambda dtype: tir.const(0, storage_dtype),
                name="bitwise_or",
            )
            w_gathered = te.compute(
                shape=(weight.shape[1], n_i32),
                fcompute=lambda j, i: reducer(
                    tir.if_then_else(
                        i * n_float_per_int + k < weight.shape[0],
                        f_scale_weight(i * n_float_per_int + k, j)
                        << (k.astype(storage_dtype) * tir.const(nbit, storage_dtype)),
                        tir.const(0, storage_dtype),
                    ),
                    axis=k,
                ),
                name="w_gathered",
            )

        return w_gathered, topi.cast(scale, "float16")

    return te_encode_sym


def decoding_func(nbit: int, storage_nbit: int):
    def te_decode_sym(data, scale):
        n_float_per_int = storage_nbit // nbit

        def f_decode_sym(i, j):
            if n_float_per_int == 1:
                data_float = tir.Cast("float16", data[i, j])
            else:
                f_convert = tir_utils._tir_packed_int_to_int_to_float(storage_nbit)
                data_float = f_convert(
                    nbit, data[i, j // n_float_per_int], j % n_float_per_int, dtype="float16"
                )

            scale_float = scale[j]
            return data_float * scale_float

        shape = (data.shape[0], data.shape[1] * n_float_per_int)
        w = te.compute(shape=shape, fcompute=f_decode_sym, name="decode")
        # Dummy transpose for FuseDecodeTranspose
        return topi.transpose(w)

    return te_decode_sym


@visitor
class FTQuantizeUpdater(QuantSpecUpdater._cls):
    def visit_call_(self, call: relax.Call):
        if call.op != tvm.ir.Op.get("relax.matmul"):
            return
        rhs = self.lookup_binding(call.args[1])
        assert rhs is not None
        if (
            rhs.op != tvm.ir.Op.get("relax.permute_dims")
            or rhs.attrs.axes is not None
            or rhs.args[0].struct_info.ndim != 2
        ):
            return

        if rhs.args[0] not in self.param_map:
            return

        param = self.param_map[rhs.args[0]]

        if call.struct_info.dtype == "float32" or rhs.struct_info.shape[-1] % 32 != 0:
            # FT requires N to be a multiple of 8
            # FT does not support fp32 output dtype
            # TODO(masahi): If `matmul(..., out_dtype="float32")` is immediately followed
            # by `cast(..., "float16")`, `matmul -> cast` can be offloaded.
            param.quant_spec = GroupQuantizationSpec(
                param.param_info.dtype,
                mode="int4",
                sym=True,
                storage_nbit=32,
                group_size=32,
                transpose=False,
            )
