function funs = geometry
  funs.create_plane = @create_plane;
  funs.get_plane_points = @get_plane_points;
  funs.intersection_between_2planes = @intersection_between_2planes;
  funs.get_line_points = @get_line_points;
  funs.intersection_between_2line = @intersection_between_2line;
  
end


function plane_param = create_plane(origin, xaxis, yaxis, zaxis=zeros(3))
  normal = cross(xaxis, yaxis);
  d = - dot(origin, normal);
  plane_param = [normal, d];
  
end

function [xx, yy, zz] = get_plane_points(origin_xy, plane_param, range)
  f = @(xx, yy) (-plane_param(4) - plane_param(1) * xx - plane_param(2) * yy) / plane_param(3);
  
  range_x = linspace(origin_xy(1),origin_xy(1) + range);
  range_y = linspace(origin_xy(2),origin_xy(2) + range);
  [xx, yy] = meshgrid(range_x, range_y);
  zz = f(xx, yy);  
  
end

function [p, v] = intersection_between_2planes(origin_x, plane0_param, plane1_param)
  cross0 = cross(plane0_param(1:3), plane1_param(1:3));
  param = [[plane0_param(2:3)];[plane1_param(2:3)]];
  value = [[-plane0_param(4)];[-plane1_param(4)]] - [[plane0_param(1)];[plane1_param(1)]] * origin_x;
  p0 = param\value;
  p = [origin_x, p0'];
  v = cross0;
  
end

function [xx, yy, zz] = get_line_points(origin, vector, range)
  t = linspace(0,range);
  point = repmat(origin, size(t)(2), 1) + repmat(t', 1, size(vector)(2)) * repmat(vector, size(vector)(2), 1);
  xx = point(:, 1); 
  yy = point(:, 2); 
  zz = point(:, 3); 
end

function p = intersection_between_2line(line0_param, line1_param)
  p0 = line0_param(1:3) ;
  v0 = line0_param(4:6) ;
  p1 = line1_param(1:3) ;
  v1 = line1_param(4:6) ;
  cross0 = cross(p1 - p0, v1) ;
  cross1 = cross(v0, v1) ;
  t = norm(cross0 / norm(cross1));
  p = p0 + t * v0;
end





