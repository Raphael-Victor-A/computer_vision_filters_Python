import cv2
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from skimage import io, filters
from skimage.filters import laplace
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte, random_noise
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, img_as_ubyte


def parse_arguments():
   parser = ArgumentParser()
   parser.add_argument("-i", "--imagem", default="exemplo.jpg", help="Imagem a ser processada")
   parser.add_argument("-r", "--ruido", default="gauss", help="Tipo de ruído. Pode ser gauss ou salt-and-pepper")
   parser.add_argument("-pr", "--pruido", default=20, type=int, help="Parâmetro do ruído. Depende do tipo do ruído")
   parser.add_argument("-s", "--suavizador", default="gauss", help="Tipo de suavizador. Pode ser mediana ou media")
   parser.add_argument("-ps", "--psuavizador", default=201, type=int, help="Parâmetro do suavizador. Depende do tipo do ruído")
   parser.add_argument("-b", "--borda", default="sobel", help="Tipo de detector de borda. Pode ser sobel ou ...")
   parser.add_argument("-pb", "--pborda", default=5, type=int, help="Parâmetro do detector de borda. Depende do tipo de detector")
   parser.add_argument("-ig", "--interface", default="sim", help="Parâmetro do suavizador. Depende do tipo do ruído")
   return parser.parse_args()

def apply_noise(image, noise_type, kernel_size=None):
   if noise_type == 'gauss':
      return random_noise(image, mode='gaussian')
   
   elif noise_type == "SaltPepper":
      row, col = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
      coords = np.array(coords)
      coords[0, coords[0] >= row] = row - 1
      coords[1, coords[1] >= col] = col - 1
      out[coords[0], coords[1]] = 1

      # Pepper mode
      num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
      coords = np.array(coords)
      coords[0, coords[0] >= row] = row - 1
      coords[1, coords[1] >= col] = col - 1
      out[coords[0], coords[1]] = 0
      return out
   
   elif noise_type == "poisson":
    if kernel_size is None:
        kernel_size = 3  # Valor padrão se o tamanho do kernel não for fornecido
    scale_factor = 10  # Ajuste este valor para aumentar o grau de ruído
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * scale_factor / kernel_size, size=image.shape) / float(vals)
    return noisy

   else:
      print(f'Noise type {noise_type} not implemented')
      return image
#-------------------------------------------------------------------------------------------------------
   
def apply_smoothing(image, smoothing_type, kernel_size):
   if smoothing_type == "gauss":
      kernel = cv2.getGaussianKernel(kernel_size, 0)   
      kernel = np.outer(kernel, kernel.transpose())
      
      return cv2.filter2D(image, -1, kernel)
   elif smoothing_type == "media":
       return cv2.blur(image, (kernel_size, kernel_size))
   
   elif smoothing_type == "mediana":
       normalized_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
       uint8_image = normalized_image.astype(np.uint8)
      #  image = image.astype(np.uint8)
      #  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       median_result = cv2.medianBlur(uint8_image, kernel_size)
       return median_result

   else:
      print(f'Smoothing type {smoothing_type} not implemented')
      return image
   
#-------------------------------------------------------------------------------------------------------

def apply_edge_detection(image, edge_detection_type):
   if edge_detection_type == "sobel":
      return filters.sobel(image)
   
   elif edge_detection_type == "canny":
      normalized_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
      uint8_image = normalized_image.astype(np.uint8)
      bordas = detectar_bordas_canny(uint8_image, 50, 150)
      return bordas
   
   elif edge_detection_type == "laplace":
      normalized_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
      uint8_image = normalized_image.astype(np.uint8)
      lap = cv2.Laplacian(uint8_image, cv2.CV_16S, ksize=15)
      
      return lap

   else:
      print(f'Tipo de detecção de borda {edge_detection_type} não implementado')
      return image
   

def detectar_bordas_canny(imagem, t_inferior, t_superior, aperture_size=3, l2_gradient=False):
      return cv2.Canny(imagem, t_inferior, t_superior, apertureSize=aperture_size, L2gradient=l2_gradient)


def generate_and_open_image(image, filename):
    # Normalize the image to the range [0, 1]
    image = exposure.rescale_intensity(image, out_range=(0, 1))
    
    # Convert the image to 8 bits and save it
    io.imsave(filename, img_as_ubyte(image))
    
    # Open and display the image
    img = plt.imread(filename)
    plt.imshow(img, cmap='gray')
    plt.show()
    
def display_images(images, titles, image_name):
   fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=True, sharey=True)
   ax = axes.ravel()
   for i, (image, title) in enumerate(zip(images, titles)):
      ax[i].imshow(image, cmap='gray')
      ax[i].axis('off')
      ax[i].set_title(title)
   fig.tight_layout()
   fig.canvas.manager.set_window_title('Resultados para imagem ' + image_name)
   plt.show()

   

def main():
   args = parse_arguments()
   while True:
      print("\nMenu:")
      print("1. Aplicar ruído")
      print("2. Aplicar suavização")
      print("3. Aplicar detecção de borda")
      print("4. Sair")
      choice = input("Escolha uma opção: ")

      if choice == "1":
         noise_type = input("Digite o tipo de ruído (gauss, SaltPepper, poisson): ")
         kernel_size = int(input("Digite o tamanho do kernel: "))
         image = io.imread(args.imagem)
         gray_image = rgb2gray(image)
         noisy_image = apply_noise(gray_image, noise_type, kernel_size)
         generate_and_open_image(noisy_image, args.imagem.split('.')[0]+"_ruido.jpg")


      elif choice == "2":
         smoothing_type = input("Digite o tipo de suavização (gauss, media, mediana): ")
         kernel_size = int(input("Digite o tamanho do kernel: "))
         image = io.imread(args.imagem)
         gray_image = rgb2gray(image)
         smoothed_image = apply_smoothing(gray_image, smoothing_type, kernel_size)
         generate_and_open_image(smoothed_image, args.imagem.split('.')[0]+"_suavizada.jpg")


      elif choice == "3":
         edge_detection_type = input("Digite o tipo de detecção de borda (sobel, canny, laplace): ")
         image = io.imread(args.imagem)
         gray_image = rgb2gray(image)
         edge_detected_image = apply_edge_detection(gray_image, edge_detection_type)
         generate_and_open_image(edge_detected_image, args.imagem.split('.')[0]+"_borda.jpg")


      elif choice == "4":
         break
      else:
         print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
   main()