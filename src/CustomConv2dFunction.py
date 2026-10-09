import numpy as np
import torch as t


# Heredamos torch.autograd.Function que ayuda a definir operaciones
# diferenciales personalizadas.
class CustomConv2dFunction(t.autograd.Function):

    # Parametros recibidos:

    # ctx
    # Es un objeto especial de pytorch que funciona como puente entre el forward y el backward
    # Permite almacenar tensores o hiperparametros para utilzar despues

    # x -> (B, C_in, H_in, W_in)
    # x -> tensor de 4D, en la que
    # B representa el tamaño del Batch(cantidad de imagenes procesadas a la vez antes de modificar el gradiente)
    # C_in representa los canales de entrada, 3 para RGB y uno para la escala de grises
    # H_in la altura de las imagenes o la cantidad de filas
    # W_in el ancho o la longitud del vector

    # weight -> (C_out, C_in, K_h, K_w)
    # weight -> tensor de 4D donde
    # weight-> Representan los filtros/kernels aprendibles que contienen:
    # C_out representa la cantidad de kernels
    # C_in reprenta la profundidad de los canales, 3 para el Kernel, y coincide con C_in de X
    # K_h, K_w representan el ancho y alto de los canales.

    # bias
    # Vector de 1D que añade un sesgo a cada canal de salida

    # stride
    # Salto, basicament como se mueve el kernel a travez del vector de entrda.

    # padding
    # Cantidad de ceros agregados alrededor de los bordes o del vector de entrada.

    # Utilizamos un metodo abstracto, por lo que directamente debemos instanciarlo con
    # el nombre de la clase.

    @staticmethod
    def forward(ctx, x, weight, bias=None, stride=1, padding=0):

        B, C_in, H_in, W_in = x.shape
        C_out, _, K_h, K_w = weight.shape

        if padding > 0:
            x_pad = t.zeros(
                (B, C_in, H_in + 2 * padding, W_in + 2 * padding),
                dtype=x.dtype,
                device=x.device,
            )

        else:
            x_pad = x

        H_out = (H_in - K_h + 2 * padding) // stride

        W_out = (W_in - K_w + 2 * padding) // stride

        out = t.zeros((B, C_out, H_out, W_out), dtype=x.dtype, device=x.device)

        for b in range(B):
            for co in range(C_out):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        w_start = j * stride

                        window = x_pad[b, : h_start + K_h, w_start : w_start + K_w]

                        val = t.sum(window * weight[co])
                        if bias is not None:
                            val += bias[co]
                        out[b, co, i, j] = val

        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding

        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        padding = ctx.padding
        stride = ctx.stride

        B, C_in, H_in, W_in = x.shape
        C_out, _, K_h, K_w = weight.shape
        _, _, H_out, W_out = grad_output.shape

        if padding > 0:
            x_pad = t.zeros(
                (B, C_in, H_in + 2 * padding, W_in + 2 * padding),
                dtype=x.dtype,
                device=x.device,
            )
        else:
            x_pad = x

        grad_weight = t.zeros_like(weight)
        grad_bias = t.zeros_like(bias) if bias is not None else None
        grad_x_pad = t.zeros_like(x_pad)

        for b in range(B):
            for co in range(C_out):
                for i in range(H_out):
                    for j in range(W_out):
                        delta_val = grad_output[b, co, i, j]
                        h_start = i * stride
                        w_start = j * stride

                        grad_x_pad[b, :, h_start + K_h, w_start : w_start + K_w] += (
                            delta_val * weight[co]
                        )

        if padding > 0:
            grad_x = grad_x_pad[
                :, :, padding : H_in + padding, padding : W_in + padding
            ]
        else:
            grad_x = grad_x_pad

        return grad_x, grad_weight, grad_bias, None, None
