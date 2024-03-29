# Currency Rate Predictor

### View live at https://currency-predictor-app.herokuapp.com/

The currency rate predictor uses freecurrency api for accessing different currencies and their conversion rates with respect to USD and predicts possible rates for next 14 days. This is a time series prediciton implemented using facebook's Prophet model. The model is deployed to a Dash based web application hosted on Heroku.  

- The web application can be viewed [here](https://currency-predictor-app.herokuapp.com/).
- Note that this project is still in progress and can be expanded to more currencies along with more accurate predictions.

Currently supported currencies are:

- Japanese Yen
- Great Britain Pound
- Australian Dollar
- Canadian Dollar
- Chinese Yen
- Hongkong Dollar
- Indian Rupee
- Singapore Dollar

Technology used:
- python
- sklearn
- pandas
- flask
- plotly dash
- jquery
- heroku

### Screenshots

Landing Page:

![Screenshot 1](/Screenshots/Confidence_removed.png)

![Screenshot 2](/Screenshots/Tabular_prediction.png)

Graph View Changes:

![Screenshot 1](/Screenshots/Initial_value_changed.png)

![Screenshot 2](/Screenshots/Region_select.png)

### Project Structre
![Screenshot 1](/Screenshots/Tree.png)

### Fowchart of Methodology:
![Screenshot 1](/Screenshots/Flowchart_of_methodology.png)
