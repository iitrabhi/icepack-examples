// This code was created by pygmsh vunknown.
p0 = newp;
Point(p0) = {-200000.0, 0, 0, 5000.0};
p1 = newp;
Point(p1) = {200000.0, 0, 0, 5000.0};
p2 = newp;
Point(p2) = {0, 0, 0, 5000.0};
p3 = newp;
Point(p3) = {0, -800000.0, 0, 5000.0};
l0 = newl;
Circle(l0) = {p0, p2, p1};
l1 = newl;
Circle(l1) = {p1, p3, p0};
ll0 = newll;
Line Loop(ll0) = {l0, l1};
s0 = news;
Plane Surface(s0) = {ll0};
Physical Line(1) = {l0};
Physical Line(2) = {l1};
Physical Surface(3) = {s0};