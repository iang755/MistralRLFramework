function [elev_map, x_vector, y_vector] = interpolate_map(elev_, map_resolution, x_vector_old, y_vector_old)

x_vector = 0:map_resolution:(round(x_vector_old(end)/10))*10; %Rounded to nearest 10m
y_vector = 0:map_resolution:(round(y_vector_old(end)/10))*10;
[Xq, Yq] = meshgrid(y_vector, x_vector);
[X, Y] = meshgrid(y_vector_old, x_vector_old);
elev_map = interp2(X, Y, elev_, Xq, Yq);