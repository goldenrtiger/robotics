flange_origin = [1021.421842,   43.072222,  742.761042];
flange_xaxis = [-0.01438783,  0.13919369, -0.99016065];
flange_yaxis = [0.59265595, 0.79875557, 0.1036748 ];

laser_origin = [1118.3517,  102.8793,  446.4760];
% laser_xaxis = [-0.5126,  0.8476,  0.1370];
laser_xaxis = [0.8584, 0.5098, 0.0571]; % yaxis, xaxis, zaxis
laser_yaxis = [-0.0215,  0.1469, -0.9889];

table_origin = [  0.,   0., 300.];
table_xaxis = [1.0,  0.0,  0.0];
table_yaxis = [0.0,  1.0, 0.0];

geometry_funs = geometry

flange_plane_param = geometry_funs.create_plane(flange_origin, flange_xaxis, flange_yaxis) ;
laser_plane_param = geometry_funs.create_plane(laser_origin, laser_xaxis, laser_yaxis) ;
table_plane_param = geometry_funs.create_plane(table_origin, table_xaxis, table_yaxis) ;
[intersection_line_p, intersection_line_v] = geometry_funs.intersection_between_2planes(laser_origin(1), laser_plane_param, table_plane_param);

[flange_xx, flange_yy, flange_zz ] = geometry_funs.get_plane_points(flange_origin(1:3), flange_plane_param, 200);
[laser_xx, laser_yy, laser_zz ] = geometry_funs.get_plane_points(laser_origin(1:3), laser_plane_param, 200);
[table_xx, table_yy, table_zz ] = geometry_funs.get_plane_points(laser_origin(1:3), table_plane_param, 200);

center_touch_point = geometry_funs.intersection_between_2line([laser_origin, laser_yaxis], [intersection_line_p, intersection_line_v]);

[line_xx, line_yy, line_zz] = geometry_funs.get_line_points(intersection_line_p, intersection_line_v, 100);
[laser_line_xx, laser_line_yy, laser_line_zz] = geometry_funs.get_line_points(laser_origin, laser_yaxis, 100);

hold on;
% surf (flange_xx, flange_yy, flange_zz);
surf (laser_xx, laser_yy, laser_zz);
legend("laser_plane");
surf (table_xx, table_yy, table_zz);
legend("table_plane");
plot3(line_xx, line_yy, line_zz, 'ro');
plot3(laser_line_xx, laser_line_yy, laser_line_zz, 'r-');
% scatter3(laser_origin(1), laser_origin(2), laser_origin(3));
scatter3(center_touch_point(1), center_touch_point(2), center_touch_point(3));
hold off;



