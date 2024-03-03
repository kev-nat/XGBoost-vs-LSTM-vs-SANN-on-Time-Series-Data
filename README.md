# Analyze and Predict Wheat Production and Demand Using Time Series Data with Different Machine Learning Algorithms

## Abstract
<p align="justify">  
Wheat is a crucial agricultural produce in the world, as it is being used as staple food around the globe in addition to feed for livestocks. Wheat production could be affected by multiple factors such as soil, weather, water, and crop management. Accurate prediction of wheat yield is essential for food security, trade, and policy making. In recent years, artificial intelligences have been widely implemented in wheat harvest prediction using remote sensing and meteorological data. These algorithms can capture the complex and nonlinear relationships between wheat yield and various input variables, and provide reliable and interpretable results. Within this paper, we review the current cutting-edge machine learning algorithms to predict the yield of a wheat production and consumption, such as LSTM (Long Short Memory Term), SANN (Seasonal Artificial Neural Network), and XGBoost (Extreme Gradient Boosting). We also discuss the advantages and challenges of using different types of remote sensing data, such as multispectral, hyperspectral, and UAV data. We highlight the potential of artificial intelligences for wheat harvest prediction in the future, and suggest some directions for further research.
</p>

## Literature Review
<p align="justify"> 
As the world faced the challenge of feeding its growing population, a wave of technological advancement swept through the global agricultural scene, as researchers explored the possibilities of machine learning to transform crop production and secure food for the future.
</p>
<p align="justify"> 
In Iran, a group of scientists set out to forecast food production, yield, and quality using state-of-the-art machine learning algorithms. They presented the adaptive network-based fuzzy inference system (ANFIS) and multilayer perceptron (MLP) models, designed to fit the specific agricultural conditions of Iran. By focusing on livestock and agricultural data, the ANFIS model, equipped with Generalized bell-shaped (Gbell) built-in membership functions, became the epitome of accuracy, providing policymakers with a reliable tool to anticipate future food production. These results not only opened the door for improved food security planning but also established a standard for algorithmic excellence in predicting agricultural outcomes.
</p>

<p align="justify"> 
Beyond borders, in the scenic landscapes of Germany, another story of innovation unfolded. A team of researchers leveraged the power of Convolutional Neural Networks (CNN) to estimate winter wheat yield with unparalleled accuracy. Weather, soil, and crop data skillfully integrated into the neural network created a model that surpassed its competitors. The researchers not only proved the superiority of the CNN model but also highlighted the importance of each feature through SHAP values. The accuracy achieved by the CNN model, even when using only the most essential features, stood as a proof of the potential of machine learning in enhancing crop yield predictions.
</p>

<p align="justify"> 
Zooming out to a bird’s eye view, a comprehensive study examined 50 machine learning-based and 30 deep learning-based papers, revealing the landscape of algorithms and features dominating the field. Artificial neural networks, convolutional neural networks, Long-Short short-term memory, and deep neural networks emerged as the pillars of innovation. Temperature, rainfall, and soil type stood as the common threads weaving through the diverse tapestry of studies. This extensive analysis not only summarized the current state of research but also paved the way for future endeavors, offering valuable suggestions for further exploration.
</p>

<p align="justify"> 
In the verdant landscapes of Bangladesh, where agriculture supports livelihoods, a vision unfolded to empower farmers with data-driven decisions. A static dataset, carefully compiled from official statistics and research councils, became the basis for decision-making. The power of machine learning algorithms—Decision Tree Learning-ID3 and K-Nearest Neighbors Regression—was harnessed to analyze historical data and current environmental factors. The result: a ranking of crops, guiding farmers to grow the most profitable crops for their specific land areas. This initiative empowered Bangladeshi farmers and showcased the transformative potential of machine learning in agricultural decision-making.
</p>

<p align="justify"> 
Meanwhile, on the Indian subcontinent, a new chapter began to unfold—aimed at supporting beginner farmers in making optimal choices. Machine learning, in conjunction with a mobile application, became the guiding light for farmers in India. Temperature, humidity, and moisture content became the metrics for informed decisions, ensuring a plentiful harvest. This initiative not only addressed the needs of novice farmers but also underscored the versatility of machine learning applications in diverse agricultural landscapes.
</p>

<p align="justify"> 
And so, across continents, the convergence of machine learning and agriculture painted a story of innovation, empowerment, and a promising future for sustainable food production.
</p>

<p align="justify"> 
This research area explores the use of various machine learning algorithms and data sources to predict food production, yield, and quality for different crops and regions.
</p>

## Methodology
<p align="justify"> 
We will be comparing the performance of different forecasting algorithms in order to find the best one. The dataset we will be using is from the USDA, which contains yearly data of various statistics for wheat, which includes production, yield, import and exports, etc.
</p>

<p align="justify"> 
The next step after data collection will be data exploration. In this step each statistics will be analyzed to identify its importance to be forecasted. In this step we have identified that wheat production, food use, and feed and residual use is important to be forecasted compared to other statistics because these statistics will help consider the supply and demand of wheat in the future.
The third step is data filtering. In this step, we filter out the data that we have not chosen. We will also convert any units used to SI units. The fourth step is to define the models used. The models that are chosen to be used are LSTM, SANN, and XGBoost.
</p>

<p align="justify"> 
The first algorithm we will be using is Long short term memory, which is a type of Recurrent Neural Network that is aimed to solve back propagation problems which includes oscillating weights and the decaying of error in proportion to the weights. In order to solve these problems, a gradient-based learning algorithm is used with the RNN architecture.
</p>

![image](https://github.com/kev-nat/XGBoost-vs-LSTM-vs-SANN-on-Time-Series-Data/assets/97384711/9cb29944-f6f7-48e4-9e31-89e1bff4de2a)

<p align="justify"> 
Above is a single LSTM cell, where ft is the forget gate, it is the input gate, Ct is the control gate, Ot is the output gate, σ is a sigmoid function and tanh is the activation function. ht and ht-1 are the current output and the previous output, respectively. Ct and Ct-1 are the current state of the cell and the previous state of the cell respectively. Since each cell's input and outputs are of the current timeframe and the past timeframe this results in each cell having some sort of memory.  From the above diagram, we can surmise the equation for the input at each gate, the output, and the states of the cell as:
</p>

- ft =(w_f ⋅ [ht-1,xt] + bf)
- it =(wi ⋅ [ht-1,xt] + bi)
- ot =(wo ⋅ [ht-1,xt] + bo)
- ct =tanh(wc ⋅ [ht-1,xt] + bc)
- ct =ft*ct-1 + it*ct
- ht=ot*tanh(ct)

Where w represents the weights at each gate and b represents the bias at each gate. 

<p align="justify"> 
XGBoost is a type of gradient boosted decision tree algorithm. Gradient boosted decision tree algorithm uses gradient boosting to combine multiple weak decision trees in order to create one strong predictor. The trees are built to minimize the previous tree's error and the final prediction is a weighted sum of all the tree's predictions. As a result of how the trees are built they are connected in series with the previous trees, this results in the algorithm being very demanding of resources, this however is what differs XGBoost to a normal gradient boosted decision tree algorithm. XGBoost creates these trees in parallel per level and it uses partial sums of trees in a level to evaluate the quality of splits in the training data. In order to evaluate the quality of splits, XGBoost uses this equation:
</p>
  
![image](https://github.com/kev-nat/XGBoost-vs-LSTM-vs-SANN-on-Time-Series-Data/assets/97384711/cf2bbc5c-3b09-4e1c-bbd9-2353e9dabdcc)

<p align="justify"> 
The first term of the equation represents the score of the left leaf after a split, the second term represents the score for the right leaf and the last term represents the original score. The term  is used in training as regularization for an additional leaf.
</p>

The values of  and indicates the quality of the structure of the tree, the smaller the values the better.

<p align="justify"> 
Seasonal artificial neural network or SANN, is first proposed to increase performance during forecasting of time series data. The architecture of this ANN model uses a variable s to indicate the number of input and output neurons. The inputs for this model are the data collected in ith seasonal period and the outputs are data collected in (i+1)th seasonal period. For example if ith seasonal period is 1 month the input is the data collected in a month and the output will be data collected in the next month. In simple terms this model uses historical data in a period to predict what is going to happen in the next period. Below is a simple diagram of the Seasonal artificial neural network:
</p>

![image](https://github.com/kev-nat/XGBoost-vs-LSTM-vs-SANN-on-Time-Series-Data/assets/97384711/bb1fc48c-9bec-44b2-a0b5-84fe9b36cfb8)

From the diagram we can formulate the following equation for the output of the neural network :

![image](https://github.com/kev-nat/XGBoost-vs-LSTM-vs-SANN-on-Time-Series-Data/assets/97384711/b149500e-583d-4a5d-9128-7b5d3a3ac0ec)

<p align="justify"> 
Where Yt+l for (l=1,2,...,s) is the future prediction for period s, Yt-i for (i=0,1,2,...,s-1) is the data captured in the previous period s, Vij for (i = 0,1,2,..., s 1; j = 1,2,...,m) are the weights for the inputs to the hidden layer, Wij for (j = 0,1,2,..., m 1; l = 1,2,...,s) are the weights for the hidden layer to the outputs, l and j are the weight biases and f is the activation function used.
</p>

<p align="justify"> 
The next step is to train the model. Before we could train our model we will split them into test and train sets. 30% will go to test and 70% will go to train. After that we proceed to training our model. In order to make a good comparison of which model is the best suited for our data, we would need to find the best parameter for our model. This is done by repeatedly changing parameters and checking the error value until a low value is found, when this happens this model will be compared to other models that have gone through these same steps. The next step is to evaluate the forecasting accuracies of each model. This is done by finding its RMSE score. Below is the equation used to find the RMSE score:
</p>

![image](https://github.com/kev-nat/XGBoost-vs-LSTM-vs-SANN-on-Time-Series-Data/assets/97384711/a01d864e-8473-40ec-a37f-35b77b83666a)

<p align="justify"> 
Where yhati are the values predicted by the algorithm, yi is the actual value and n is the total number of measurements. This RMSE equation will be used on the test set of data, as this will represent the real world performance because we are predicting values that are out of the training set.
</p>

## Results and Analysis

### Dataset Visualization and Analysis
<p align="justify"> 
Wheat production in America has been growing since 1950 until the 1980s where it peaked at 752.49 million metric tons of wheat in 1982 and slowly dropping since 1990. We could see that one of the factors causing the drop in production is the diminishing growth in the demand. As we can see in the figure below, the demand of wheat in the feed and residual use dropped drastically since 1990 while the demand of wheat in food use got stagnant since the 2000s.
</p>

<p align="justify"> 
Looking at the data that we get from the federal government of the United States of America, we can see that in the last 10 years people's food use tends to be constant, not only in the last 10 years but since the 1950s. We can see that there is no drastic increase in the number of wheat usage from year to year, all of them are in the range of 150 - 250 million metric tons. This indicates that wheat is not the main staple ingredient in the American diet, we know that not all Americans eat bread but many of them prefer to eat potatoes or any kind of food.
</p>

![image](https://github.com/kev-nat/XGBoost-vs-LSTM-vs-SANN-on-Time-Series-Data/assets/97384711/0403eea3-2bc2-44de-9f24-67d0ec7b90e5)

### Prediction Result
<p align="justify"> 
With the USA wheat production and consumption data from 1950 to 2022 we use LSTM, SANN, and XGBoost machine learning algorithms to predict the last 10 years of USA wheat production and consumption.
</p>

<details>
<summary> Wheat Production </summary>
<p align="justify"> 
Based on the figure below we could see that out of the 3 algorithms that we use XGBoost produces the closest prediction to the real data compared to LSTM or SANN. This could be attributed to the fact that the USA wheat production data is fluctuating significantly.
</p>
  
Wheat production prediction using LSTM algorithm (RMSE = 81.5489)
![image](https://github.com/kev-nat/XGBoost-vs-LSTM-vs-SANN-on-Time-Series-Data/assets/97384711/a013d9b1-f993-403e-b50d-6f3c417c8924)

Wheat production prediction using SANN algorithm (RMSE = 80.39)
![image](https://github.com/kev-nat/XGBoost-vs-LSTM-vs-SANN-on-Time-Series-Data/assets/97384711/5af6f99d-40f1-42d7-988a-7b7def001b8e)

Wheat production prediction using XGBoost algorithm (RMSE = 13.63)
![image](https://github.com/kev-nat/XGBoost-vs-LSTM-vs-SANN-on-Time-Series-Data/assets/97384711/7076ccbd-1963-431a-9886-abbfe9299ad2)
</details>

<details>
<summary> Food Use </summary>
<p align="justify"> 
Based on the figure below we could see that out of the 3 algorithms that we use SANN produces the closest prediction to the real data compared to LSTM or XGBoost. But based on the graph we could see that the SANN prediction graph is making a lot of fatal errors, compared to LSTM and XGBoost that have bigger RMSE but the individual error is small. And between LSTM and XGBoost, LSTM yields a lower RMSE with RMSE = 8.6221, this could be attributed due to the data having less fluctuation and more constant.
</p>

Food Use prediction using LSTM algorithm (RMSE = 7.0488)
![image](https://github.com/kev-nat/XGBoost-vs-LSTM-vs-SANN-on-Time-Series-Data/assets/97384711/d3361166-f277-4788-be39-4c3e453180d1)

Food use prediction using SANN algorithm (RMSE = 20.52)
![image](https://github.com/kev-nat/XGBoost-vs-LSTM-vs-SANN-on-Time-Series-Data/assets/97384711/11c70e1c-cb3a-4cd8-86b6-09eb475d9b63)

Food use prediction using XGBoost algorithm (RMSE =14.47)
![image](https://github.com/kev-nat/XGBoost-vs-LSTM-vs-SANN-on-Time-Series-Data/assets/97384711/677c96fc-8ce1-4a6c-84c4-d1f89b1fa1be)
</details>

<details>
<summary> Feed and Residual Use </summary>
<p align="justify"> 
Based on the figure below we could see that out of the 3 algorithms that we use XGBoost produces the closest prediction to the real data compared to LSTM or SANN. This could be attributed to the fact that the USA feed and residual use data is similar to USA wheat production data with a significant quantity of pronounced peak and valley in the data.
</p>

Feed and residual use prediction using LSTM algorithm (RMSE = 25.0403)
![image](https://github.com/kev-nat/XGBoost-vs-LSTM-vs-SANN-on-Time-Series-Data/assets/97384711/936836e0-2ddc-40f2-9cf5-4605bd7f9f74)

Feed and residual use prediction using SANN algorithm (RMSE = 74.245)
![image](https://github.com/kev-nat/XGBoost-vs-LSTM-vs-SANN-on-Time-Series-Data/assets/97384711/ef73ab91-6731-4ab6-861c-f0fdce875acc)

Feed and residual use prediction using XGBoost algorithm (RMSE = 3.18)
![image](https://github.com/kev-nat/XGBoost-vs-LSTM-vs-SANN-on-Time-Series-Data/assets/97384711/6b5dd085-aefc-42c0-925f-acf16baf15c2)
</details>

<details>
<summary> Root Mean Squared Error (RMSE) </summary>
<p align="justify"> 
Based on the RMSE graph we could see that XGBoost has the best overall performance on our data while LSTM took the 2nd place and SANN as the worst performance.

In the production and feed and residual use prediction XGBoost performs the best with RMSE = 13.63 for the production use and RMSE = 3.18 for feed and residual use, compared to LSTM that only have RMSE = 80.3950 for production and RMSE = 24.9607 for feed and residual use. This could be attributed to the high fluctuation in production and feed and residual use data, XGBoost could yield a low RMSE compared to LSTM because XGBoost doesn't fully ignore outliers. But for a stabler data such as the food use data, we could see that XGBoost yields a higher RMSE with RMSE = 14.47 compared to LSTM with RMSE = 8.6221, while for SANN it could yield better RMSE compared to XGBoost and LSTM with RMSE = 7.158 in a stabler data but the prediction often makes a fatal mistake making it not reliable for prediction. 
</p>

The RMSE of three algorithms predicting USA wheat production for the years 2012-2022.
![image](https://github.com/kev-nat/XGBoost-vs-LSTM-vs-SANN-on-Time-Series-Data/assets/97384711/f5cd780a-8fd2-48c1-a406-45383cb78de1)

The RMSE of three algorithms predicting USA food use for the years 2012-2022.
![image](https://github.com/kev-nat/XGBoost-vs-LSTM-vs-SANN-on-Time-Series-Data/assets/97384711/d480d43e-4be3-44ba-be7d-4271aad8a312)

The RMSE of three algorithms predicting USA feed and residual use for the years 2012-2022.
![image](https://github.com/kev-nat/XGBoost-vs-LSTM-vs-SANN-on-Time-Series-Data/assets/97384711/c1f75253-513d-4e1b-8f01-c7b8f7c2f410)
</details>

### Conclusion
<p align="justify"> 
From the wheat production analysis that we have carried out using 3 types of machine learning algorithms, namely, LSTM (Long Short Term Memory), SANN (Seasonal Artificial Neural Network), and XGBoost (Extreme Gradient Boosting).

We can conclude that for cases with our data that have a lot of fluctuations, the XGBoost algorithm has higher accuracy compared to LSTM and SANN. This can happen because XGBoost works by iteratively building a new tree based on predecessor learners' residuals. As outliers have much larger residuals than non-outliers, the algorithm may disproportionately focus its attention on those points. Hence, the presence of outliers in the dependent regressor variable can make XGBoost sensitive to outliers.

However, XGBoost has a weakness, namely that if it is faced with a dataset that tends to be constant as we can see in the food use case, we can see that the predictions tend to be flat when compared to the LSTM algorithm.
</p>

### References
[1]J. Brownlee, “Time Series Prediction with LSTM Recurrent Neural Networks in Python with Keras,” Machine Learning Mastery, Apr. 26, 2019. https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/.

[2]dmlc, “dmlc/xgboost,” GitHub, Oct. 25, 2019. https://github.com/dmlc/xgboost.

[3]Pantazi, X.E. & Moshou, Dimitrios & Alexandridis, Thomas & Whetton, Rebecca & Mouazen, Abdul. (2016). Wheat yield prediction using machine learning and advanced sensing techniques. Computers and Electronics in Agriculture. 121. 57-65. 	10.1016/j.compag.2015.11.018.

[4]“USDA ERS - Wheat Data,” Usda.gov, 2018. https://www.ers.usda.gov/data-products/wheat-data/.

[5]S. Nosratabadi, S. F. Ardabili, Z. Lakner, C. Makó, and A. Mosavi, “Prediction of Food Production Using Machine Learning Algorithms of Multilayer Perceptron and ANFIS,” SSRN Electronic Journal, 2021, doi: https://doi.org/10.2139/ssrn.3836565.

[6]Amit Kumar Srivastava et al., “Winter wheat yield prediction using convolutional neural networks from environmental and phenological data,” Scientific Reports, vol. 12, no. 1, Feb. 2022, doi: https://doi.org/10.1038/s41598-022-06249-w.

[7]T. van Klompenburg, A. Kassahun, and C. Catal, “Crop yield prediction using machine learning: A systematic literature review,” Computers and Electronics in Agriculture, vol. 177, p. 105709, Oct. 2020, doi: https://doi.org/10.1016/j.compag.2020.105709.

[8]A. Talaviya, “Crop Yield Prediction Using Machine Learning And Flask Deployment,” Analytics Vidhya, June. 17, 2023. https://www.analyticsvidhya.com/blog/2023/06/crop-yield-prediction-using-machine-learning-and-flask-deployment/#:~:text=Crop%20field%20 prediction%20is%20an.

[9]“Agricultural production output prediction using Supervised Machine Learning techniques | IEEE Conference Publication | IEEE Xplore,” ieeexplore.ieee.org. https://ieeexplore.ieee.org/abstract/document/8016196/.  (accessed Dec. 16, 2023).

[10]M. Kalimuthu, P. Vaishnavi, and M. Kishore, “Crop Prediction using Machine Learning,” 2020 Third International Conference on Smart Systems and Inventive Technology (ICCSIT), Aug. 2020, doi: https://doi.org/10.1109/icssit48917.2020.9214190.

[11]F. Marini, “Artificial neural networks in foodstuff analyses: Trends and perspectives A review,” Analytica Chimica Acta, vol. 635, no. 2, pp. 121–131, Mar. 2009, doi: https://doi.org/10.1016/j.aca.2009.01.009.‌

[12]HAMZACEBI, C. (2008). Improving artificial neural networks’ performance in seasonal time series forecasting. Information Sciences, 178(23), 4550–4559. https://doi.org/10.1016/j.ins.2008.07.024.

[13]Luo, J., Zhang, Z., Fu, Y., & Rao, F. (2021). Time series prediction of COVID-19 transmission in America using LSTM and XGBoost algorithms. Results in Physics, 27, 104462. https://doi.org/10.1016/j.rinp.2021.104462.

[14]Ratnadip Adhikari, & R. K. Agrawal. (2013). An Introductory Study on Time Series Modeling and Forecasting. Doi:https://doi.org/10.48550/arXiv.1302.6613. 
