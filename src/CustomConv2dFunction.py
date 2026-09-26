import torch as t
import numpy as np


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
    # H_in la altura de las imagenes o la cantidad de filas, W_in el ancho o la longitud del vector

    # weight -> (C_out, C_in, K_h, K_w)
    # weight -> tensor de 4D donde
    # weight-> Representam los filtros/kernels aprendibles que contienen:
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
        C_out, C_in, K_h, K_w = weight.shape

        if padding > 0:
            x_pad = np.zeros(x.shape)
