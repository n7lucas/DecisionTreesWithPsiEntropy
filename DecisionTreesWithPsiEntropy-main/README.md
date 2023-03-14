<h1>Decision trees with &#x3A8 entropy</h1>


<h2>Introduction</h2>

<p>Main function for using entropy classifier</p>

<h2>File Structure</h2>

<p>The code is responsible for calling the necessary functions to create a classifier model with the Entropic Classifier</p>

<p><strong>Methods:</strong></p>
<ul>
 <li> mytrain_test_split(X,y, train_size): function developed to separate the data according to the size of the training data.</li>

<li>plot_clf(model, y_test, X_test, activationname): Function responsible for plotting the separations performed by the classifier with two-dimensional data</li>

<h2>Usage: </h2>

<p>To use the classifier, it is first necessary to import data, in this example below we are using the moon_dataset provided by the sklearn library</p>>
<p>train_val = 70/100
X, y = make_moons(n_samples=100, shuffle=True, noise=0.1, random_state=1)</p>

<p> To separate the training and test data we will use the bla function, and as it only accepts the pandas dataframe format, the data X, and y originally distributed in a numpy array are converted to a pandas dataset and then converted again to the format from numpy array</p>
<p>
X = pd.DataFrame(X)
y = pd.DataFrame(y, columns=['classe'])
X_train, X_test, y_train, y_test = mytrain_test_split(X,y, train_val)
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()</p>


<p>We instantiate the P2 class that has the functions for creating the quadratic classifier passing the activation function and optimization method</p>
<p>model = P2(activation='heaviside', optimizer="montecarlo") </p>

<p>We call the run function providing the data as input, the data returned by the function is only for the data plot</p>
<p>best_params_calc,list_S, list_r = model.run(X_train, y_train, X_train, y_train)</p>
<p>Com o modelo criado é possivel acessar algumas informações sobre o classificador</p>

<p>Retornar nodes gerados pela arvore</p>
 ``print("Hello World!")``
<p> nodes = model.tree.nodes</p>

<p>Retornar uma lista com os nodes que um determinado input x percorreu na arvore até chegar a uma folha</p>
<p>
x = np.array([0.954831, -0.134719]) #Estou passando 1 elemento da base de dados moon-dataset na qual é um conjunto de coordendas
pathList = model1.path(x, 0) # Além do elemento da base de dados, é necessario passar o treshhold, como estamos utilizando heaviside este valor é 0
</p>

<p>É possivel retornar somente o node final na qual o input x com a função leaf() demonstrada abaixo</p>
leaf = model1.leaf(arr,0)

<p> Retornar lista de todos elementos que passarm por um determinado nó é possivel acessando o atributo C_k, na qual cada node armazena a informação de cada node que passou por ele, abaixo um exemplo acessando o C_k do primeiro node</p>
<p>ck = model.tree.nodes[1].C_k</p>

<p> Para verificar as separações geradas pelo classificador é possivel utilizar o método plot_clf(), que funciona com dados de até 2 dimensões, o resultado para o moons-dataset é mostrado abaixo com os dados de in-sample</p>
<img src="plot.png">


