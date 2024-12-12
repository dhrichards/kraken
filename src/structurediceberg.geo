
top = 0.4;
bot = -0.5;
crack_x = 1.5;
L = 2;

Point(1) = {0, bot, 0, 1.0};
Point(2) = {0, top, 0, 1.0};
Point(3) = {L, top, 0, 1.0};
Point(4) = {L, bot, 0, 1.0};

Point(5) = {crack_x, bot, 0, 1.0};
Point(6) = {crack_x, top, 0, 1.0};

Line(1) = {1, 2}; // left
Line(2) = {2, 6}; // top left
Line(3) = {3, 4}; // right
Line(4) = {5, 1}; // bottom left

Line(5) = {6, 3}; // top right
Line(6) = {4, 5}; // bottom right

Line Loop(7) = {2, 5, 3, 6, 4, 1};

Plane Surface(8) = {7};

Transfinite Surface {8} = {2, 3, 4, 1};

Transfinite Line {1, -3} = 10 Using Progression 0.8; // left right
Transfinite Line {2, -4} = 20 Using Progression 0.8;
Transfinite Line {6,-5} = 10 Using Progression 0.8;

Recombine Surface {8};

