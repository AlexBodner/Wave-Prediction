El script read_and_transform2world.py levanta las profundidades de la rosbag, las transformaciones y los datos del filtro de madgwick sobre la imu
y transforma los puntos de coordenadas de la camara a coordenadas relativas a coordenadas de mundo usando la transformacion dada en quaterniones por el madgwick.

Por ahora solo ploteo la transformacion sobre una imagen seleccionada en la linea 168
Se corre con:

python3.10 read_and_transform2world.py