



## Generar grillas y plano medio

descargar el bag y poner en el directorio datasets. 


Para generar las grillas y el plano medio a partir de la rosbag, ejecutar:

```bash

source /opt/ros/humble/setup.bash

python3.10 process_images_for_nn.py
```

Esto creará el directorio `numpy_grids` con los archivos de grillas (`grid_{timestamp}.npy`) y el plano medio (`mean_plane.npz`).

## Visualizar grillas y plano medio

Para visualizar las grillas y el plano medio junto con la imagen de profundidad más cercana, ejecutar:

```bash
python3.10 visualize_grids.py
```

Esto abrirá una ventana para cada grilla y mostrará la imagen de profundidad correspondiente.

## Entrenar el modelo con

```bash
python3.10 main/train_model.py
```

- La arquitectura se determina en main/grid_cnn.py
- Los archivos de build_grid_dataset.py y grid_dataset sirven para juntar las correspondencias de imagenes y target y luego crear el dataset en formato torch.


