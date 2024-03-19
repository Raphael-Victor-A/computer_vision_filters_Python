Esboço de código em python para aplicar alguns ruídos, filtros
de suavização e detectores de borda. 

Testado com

- Ubuntu 20.04
- conda 4.10.3
- python 3.9.7

Autor: Hemerson Pistori (pistori@ucdb.br)

------------------------------------------------------
Preparar o ambiente e instalar dependências. 
------------------------------------------------------
Comando úteis:
$ conda create -n vc scikit-image opencv=4.5.4 matplotlib=3.5.1
$ conda env list
$ conda activate vc
$ conda list
  Remove tudo para instalar de novo:
$ conda remove --name vc --all

------------------------------------------------------
Executar o código para uma imagem
------------------------------------------------------
Comandos úteis:
$ conda activate vc
$ python --version
$ python ruidoSuaveBordas.py -i exemplo.jpg -r gauss -pr 20 -s gauss -ps 101 -b sobel

------------------------------------------------------
Resultados esperados
------------------------------------------------------
- Mostrará na tela a imagem original juntamente com
  as imagens com ruído, suavizada e bordas
- Salvará as imagens resultantes em disco na
  mesma pasta que a imagem original mas com nome
  modificado 
