<h1>Entropy Classifier</h1>


<h2>Introduction</h2>
<p>EntropyClassifier is a Python class that implements a decision tree-based binary classifier using the concept of entropy. It is designed to be flexible and customizable, allowing users to specify various hyperparameters such as the optimizer algorithm, number of iterations, and activation function.</p>


<h2>File Structure</h2>

<p>The code consists of two classes: <span style="background-color: #f2f2f2">"EntropyClassifier"</span> and "P2". The "EntropyClassifier". The class EntropyClassifier contains the "generic" methods that will be used by any &#x3A8 function and optimization method, while class P2 contains the necessary modifications in each function used to work according to quadratic equations</p>

<h2>Class Structure</h2>

<h4>EntropyClassifier:</h4>

As previously mentioned, the "EntropyClassifier" class has the general methods of the tree, not being accessed by the end user, but by the class that changes these methods to work according to a PSI equation (in this case quadratic equations): 

<p><strong>Hyperparameters:</strong></p>
<ul>
    <li>optimizer: The optimization algorithm to use. Currently, only the Monte Carlo algorithm is supported. Default is 'monte carlo'.</li>
    <li>nb_iter: The number of iterations to run the optimizer for. Default is 10000.</li>
    <li>nb_leaf: Minimum percentage of data allowed in each leaf node to be able to separate.</li>
    <li>nb_branch_max: The maximum number of nodes allowed for the tree.</li>
    <li>activation: The activation function to use for calculating node values. Currently, only the Heaviside step function is supported. Default is 'heaviside'.</li>
</ul>


<p><strong>Methods:</strong></p>
<ul>
 <li>__init__(self, index, left=None, right=None, children = [], parent = []): Responsible for creating the node, receives the index value of the node, as well as its left and right children, a list for its children and another for its parent node.</li>

<li>_normalization(self, a): method to normalize the vector <strong>a</strong></li>

<li>_norm_vector (self,a): Another function to normalize <strong>a</strong>.</li>

<li>_Line(self,X,a): Calls functions for calculating psi</li>
         
<li>_parcial_entropy(self,f_pos, f_neg): Calculates Partial entropies (S+ and S-)</li>
<li>_total_entropy_heavi(self,X,a): Calculates the total entropy according to the heaviside functions</li>
<li> __dominant_class(self,D_pos, D_neg): Returns the predominant class according to a given set of data, (used to determine the predominant class in each node of the tree)</li>
<li>__plot_accuracy_brench (self,best_params): Plotting stuff</li>
<li>__grafico_entropia_low (self,list_S, list_r): Plotting stuff</li>
<li>__grafico_vetora(self,plot_a, plot_b, plot_c, list_r): Plotting stuff</li>
<li>__monte_carlo(self,X): Monte Carlo function, basically it calls the _Line function the number of times passed to the function and returns the parameters found in the iteration with the best total entropy value</li>
<li>__readable_leaf(self,Tree,nb_branch): It goes through the nodes of the tree and returns the node with the highest entropy value to make a new split</li>
<li>__stop_leaf(self, leaf): Checks if a node is validated for split, with the condition that the data present in the node still have distinct classes and the number of branches is less than the defined limit</li>
<li>_Tree(self, X, y):Function responsible for creating the tree, and distributing information on each Node</li>
<li>predict(self,x_teste, threshold): function used to predict new parameters, receives a dataset with only attributes and returns a list with the classes of each element</li>
<li>__find_label(self,x,threshold): Used to traverse the tree and stop when it finds a leaf</li>
<li>__next_node(self,node,row,threshold): According to the supplied input (x) check if the result of the psi function will direct this input to the next right or left node of the tree</li>
<li>_calcule_psi(self,row,a): Returns the value of the PSI function according to the provided input</li>
<li>run(self,X, y, x_test, y_test): Function used to create the model, it calls the _Tree function, and provides the parameters passed as data, the minimum percentage of data per node and the maximum number of branches allowed for tree generation</li>
</ul>

<h4>P2:</h4>

<p><strong>Attributes:</strong></p>
<p>It has no attributes, basically it is a class that subscribes to some methods of class <strong>EntropyClassifier</strong></p>

<p><strong>Methods:</strong></p>
<ul>
    <li>_Line(self,X,a): Overwrites the Line function with the information to perform quadratic calculations</li>
    <li>_random_a(self,X): Subscribes the _random_a function to generate the vector a for quadratic functions</li>
    <li>_gerar_reta_p2(self,X,a): Performs the calculation of the quadratic function</li>
    <li>_total_entropy_heavi(self,X,a): Subscribes to the _total_entropy_heavi function to perform quadratic calculation</li>
    <li>_calcule_psi(self,row,aç): Function that overwrites the _calculate_psi function, used to calculate the Psi function to make predictions for quadratic equations</li>
</ul>




