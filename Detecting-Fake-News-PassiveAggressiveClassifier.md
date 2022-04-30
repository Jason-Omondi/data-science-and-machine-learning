# Fake News Detection Project 


```python
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import plotly.graph_objects as go
import itertools
```


```python
df_news = pd.read_csv('news.csv')
df_news.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>title</th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8476</td>
      <td>You Can Smell Hillary’s Fear</td>
      <td>Daniel Greenfield, a Shillman Journalism Fello...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10294</td>
      <td>Watch The Exact Moment Paul Ryan Committed Pol...</td>
      <td>Google Pinterest Digg Linkedin Reddit Stumbleu...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3608</td>
      <td>Kerry to go to Paris in gesture of sympathy</td>
      <td>U.S. Secretary of State John F. Kerry said Mon...</td>
      <td>REAL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10142</td>
      <td>Bernie supporters on Twitter erupt in anger ag...</td>
      <td>— Kaydee King (@KaydeeKing) November 9, 2016 T...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>4</th>
      <td>875</td>
      <td>The Battle of New York: Why This Primary Matters</td>
      <td>It's primary day in New York and front-runners...</td>
      <td>REAL</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6903</td>
      <td>Tehran, USA</td>
      <td>\nI’m not an immigrant, but my grandparents ...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7341</td>
      <td>Girl Horrified At What She Watches Boyfriend D...</td>
      <td>Share This Baylee Luciani (left), Screenshot o...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>7</th>
      <td>95</td>
      <td>‘Britain’s Schindler’ Dies at 106</td>
      <td>A Czech stockbroker who saved more than 650 Je...</td>
      <td>REAL</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4869</td>
      <td>Fact check: Trump and Clinton at the 'commande...</td>
      <td>Hillary Clinton and Donald Trump made some ina...</td>
      <td>REAL</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2909</td>
      <td>Iran reportedly makes new push for uranium con...</td>
      <td>Iranian negotiators reportedly have made a las...</td>
      <td>REAL</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_news.shape
```




    (6335, 4)




```python
df_news.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6335 entries, 0 to 6334
    Data columns (total 4 columns):
     #   Column      Non-Null Count  Dtype 
    ---  ------      --------------  ----- 
     0   Unnamed: 0  6335 non-null   int64 
     1   title       6335 non-null   object
     2   text        6335 non-null   object
     3   label       6335 non-null   object
    dtypes: int64(1), object(3)
    memory usage: 198.1+ KB
    


```python
df_news.tail(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>title</th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6325</th>
      <td>8411</td>
      <td>Will the Media Reset After the Election or Are...</td>
      <td>Written by Peter Van Buren   venerable New Yor...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>6326</th>
      <td>6143</td>
      <td>DOJ COMPLAINT: Comey Under Fire Over Partisan ...</td>
      <td>DOJ COMPLAINT: Comey Under Fire Over Partisan ...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>6327</th>
      <td>3262</td>
      <td>GOP Senator David Perdue Jokes About Praying f...</td>
      <td>The freshman senator from Georgia quoted scrip...</td>
      <td>REAL</td>
    </tr>
    <tr>
      <th>6328</th>
      <td>9337</td>
      <td>Radio Derb Is On The Air–Leonardo And Brazil’s...</td>
      <td></td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>6329</th>
      <td>8737</td>
      <td>Assange claims ‘crazed’ Clinton campaign tried...</td>
      <td>Julian Assange has claimed the Hillary Clinton...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>6330</th>
      <td>4490</td>
      <td>State Department says it can't find emails fro...</td>
      <td>The State Department told the Republican Natio...</td>
      <td>REAL</td>
    </tr>
    <tr>
      <th>6331</th>
      <td>8062</td>
      <td>The ‘P’ in PBS Should Stand for ‘Plutocratic’ ...</td>
      <td>The ‘P’ in PBS Should Stand for ‘Plutocratic’ ...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>6332</th>
      <td>8622</td>
      <td>Anti-Trump Protesters Are Tools of the Oligarc...</td>
      <td>Anti-Trump Protesters Are Tools of the Oligar...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>6333</th>
      <td>4021</td>
      <td>In Ethiopia, Obama seeks progress on peace, se...</td>
      <td>ADDIS ABABA, Ethiopia —President Obama convene...</td>
      <td>REAL</td>
    </tr>
    <tr>
      <th>6334</th>
      <td>4330</td>
      <td>Jeb Bush Is Suddenly Attacking Trump. Here's W...</td>
      <td>Jeb Bush Is Suddenly Attacking Trump. Here's W...</td>
      <td>REAL</td>
    </tr>
  </tbody>
</table>
</div>



## Missing Values


```python
df_news.isnull()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>title</th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6330</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6331</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6332</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6333</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6334</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>6335 rows × 4 columns</p>
</div>




```python
df_news.isnull().sum()
```




    Unnamed: 0    0
    title         0
    text          0
    label         0
    dtype: int64




```python
z = df_news.label.value_counts()
```


```python
plot = go.Figure(data = [go.Bar(x=['Real','Fake'], y=z, text=z)])
plot.show()
```


<div>                            <div id="4aea1d65-bb36-4bb0-8ad7-eec5ec86011c" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("4aea1d65-bb36-4bb0-8ad7-eec5ec86011c")) {                    Plotly.newPlot(                        "4aea1d65-bb36-4bb0-8ad7-eec5ec86011c",                        [{"text":[3171.0,3164.0],"x":["Real","Fake"],"y":[3171,3164],"type":"bar"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('4aea1d65-bb36-4bb0-8ad7-eec5ec86011c');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


Split the data into test and train


```python
X_train,X_test,y_train,y_test=train_test_split(df_news['text'], df_news.label, test_size=0.2, random_state=7)
```


```python
X_train
```




    6237    The head of a leading survivalist group has ma...
    3722    ‹ › Arnaldo Rodgers is a trained and educated ...
    5774    Patty Sanchez, 51, used to eat 13,000 calories...
    336     But Benjamin Netanyahu’s reelection was regard...
    3622    John Kasich was killing it with these Iowa vot...
                                  ...                        
    5699                                                     
    2550    It’s not that Americans won’t elect wealthy pr...
    537     Anyone writing sentences like ‘nevertheless fu...
    1220    More Catholics are in Congress than ever befor...
    4271    It was hosted by CNN, and the presentation was...
    Name: text, Length: 5068, dtype: object




```python
y_train
```




    6237    FAKE
    3722    FAKE
    5774    FAKE
    336     REAL
    3622    REAL
            ... 
    5699    FAKE
    2550    REAL
    537     REAL
    1220    REAL
    4271    REAL
    Name: label, Length: 5068, dtype: object



## Initialize a TfidVectorizer


```python
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
```

Fitting and Transforming both training and testing data


```python
tfidf_train=tfidf_vectorizer.fit_transform(X_train) 
tfidf_test=tfidf_vectorizer.transform(X_test)
```

PassiveAggressiveClassifier


```python
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
```




    PassiveAggressiveClassifier(max_iter=50)



## Prediction on the testing data


```python
y_pred=pac.predict(tfidf_test)
```


```python
score=accuracy_score(y_test,y_pred)
print(f'Accuracy Percentage: {round(score*100,3)} %')
```

    Accuracy Percentage: 92.897 %
    

## Confusion Matrix


```python
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
```




    array([[590,  48],
           [ 42, 587]], dtype=int64)



Creating a Report 


```python
print('\n clasification report:\n',classification_report(y_test,y_pred))
```

    
     clasification report:
                   precision    recall  f1-score   support
    
            FAKE       0.93      0.92      0.93       638
            REAL       0.92      0.93      0.93       629
    
        accuracy                           0.93      1267
       macro avg       0.93      0.93      0.93      1267
    weighted avg       0.93      0.93      0.93      1267
    
    
