Input Data:

$T=\lbrace t_1, ..., t_7 \rbrace$ — Set of trees.

$Y(t)=\lbrace y_{t1}, ..., y_{tn_t} \rbrace$ — Set of years fow which the meassurements for the tree $t$ are available,  $t \in T$

$Y=\bigcup_{t \in T} Y(t)$ — Set of all years for witch the meassurements are available.

$T(y)=\lbrace t_{y1}, ..., t_{ym_y} \rbrace$ — Set of trees for which the meassurements for the year $y$ are availvable, $y \in Y$

$\left (T \equiv \bigcup_{y \in Y} T(y) \right )$

Normalization procedure description:

$N$ — Number of cells for tracheid normalization.

$e^{raw}(t,y)=\lbrace e^{raw}_1, ..., e^{raw}_{\varepsilon}\rbrace$ — Raw tracheid data where:

$e^{raw}_k = e^{raw}_k(t,y) \in \lbrace d^{raw}_k, c^{raw}_k\rbrace$

$d^{raw}_k=d^{raw}_k(t,y)$ — Diameter of the $k^{th}$ cell in raw tracheid

$c^{raw}_k=c^{raw}_k(t,y)$ — Cell wall thickness of the $k^{th}$ cell in raw tracheid

$k=\overline{1,\varepsilon}$, $t\in T, y\in Y(t)$

$e^* = \lbrace\underbrace{e^{raw}_1,...,e^{raw}_1}_{N},\underbrace{e^{raw}_2,...,e^{raw}_2}_{N}, ..., \underbrace{e^{raw}_{\varepsilon},...,e^{raw}_{\varepsilon}}_{N}\rbrace$ — Intermediate sequence.


$e = \lbrace e_1, ..., e_N\rbrace$ — Normilized to N cells tracheid data, where 

$$e_i = \frac{1}{\varepsilon} \sum_{j=\varepsilon \cdot (i-1)+1}^{\varepsilon \cdot i}e^{*}_j, i=\overline{1, N}$$

Using this procedure, we obtain:

$d = \lbrace d_1, ..., d_N\rbrace$ — Normilized to N cells data about tracheid cell diameters

$c = \lbrace c_1, ..., c_N\rbrace$ — Normilized to N cells data about tracheid cell wall thicknesses



Clusturing objects description:

$R(t,y) =d \cup c = \lbrace d_1, ... , d_{N}, c_1, ..., c_{N}\rbrace$ — Tracheid normalized to $N$ cells. Where:

$d_i=d_i(t,y)$ — Diameter of the $i^{th}$ cell in normalized tracheid

$c_i=c_i(t,y)$ — Cell wall thickness of the $i^{th}$ cell in normalized tracheid

$i=\overline{1,N}$, $t\in T, y\in Y(t)$


Clustering Methods description:


Method A:

1. $$ R^A(y)=\frac{1}{\left| T(y) \right|}\sum_{t\in T(y)}R(t,y), y\in Y $$

2. $$R_{mean}^A=\frac{1}{\sum_{t\in T}\left| Y(t)\right|}\sum_{t\in T}\sum_{y\in Y(t)}R(t,y)$$

3. $$O_A(y)=\frac{R^A(y)}{R_{mean}^A}, y\in Y$$

Method B:

1. $$R^B(t)=\frac{1}{\left| Y(t) \right|}\sum_{y\in Y(t)}R(t,y), t\in T$$

2. $$o_B(t,y)=\frac{R(t,y)}{R^B(t)}, t\in T, y\in Y(t)$$

3. $$O_B(y)=\frac{1}{\left| T(y) \right|}\sum_{t\in T(y)}o_B(t,y), y\in Y$$