<h1>Detailed evaluation metrics</h1>
<img src="https://github.com/user-attachments/assets/686f1189-9a6b-421e-8fbc-5942de634158">

Model : Linear regression
<h2>Features</h2>
Features used : <br>
- <b>Property type</b>: house, villa, triplex, ... (from immoweb scraping) <br>
- <b>Living area</b>: the stringest correlation with the price (from immoweb scraping) <br>
- <b>Surface of the plot</b> (from immoweb scraping) <br>
- <b>Building condition</b> (from immoweb scraping) <br>
- <b>Swimming pool</b> (from immoweb scraping) <br>
- <b>District</b>: broader than the zip code but smaller than the province (statbel and matched with zip codes thanks to a CSV file found with zip codes and associated refnis code) <br>
- <b>Mean income</b> based on the zip code (statbel)  <br>
- <b>Median price</b> of apartments or houses (statbel: median price split for houses and apartments per district - I made a new columns with the median price based on the property type (house or apartment))

![correlation-heatmap-two](https://github.com/user-attachments/assets/fc7c300a-e312-42ac-8376-200f69a87094)

I used the features with the best correlation from the immoweb dataset except when there was a too strong correlation with another parameter (like bedrooms and living area).<br>
I found the added features on statbel.<br>
I tested a lot of other parameters. Some had also a good correlation with the price like number of transactions (apartments and houses sold per year and per district), population per surface per district, number of parcels built per district,... But these features were too correlated with the mean income or median price which has a better correlation with the price. <br>
I tested also other parameters but the correlation was not good.<br>
I scraped immoweb again to get other information: construction year, parking and number of bathrooms but the corrlation was not good or they were too correlated with another column. <br>
So I used the previous dataset I had since it was bigger.

<h2>Dataset</h2>
The final dataset shape: (10535, 66)
<h3>Outliers</h3>
I removed houses more expensive than 2 500 000 et apartments more expensive than 1 000 000. I got a better result with this theshold. I removed a few other outliers (based on living area, ...).<br>
I also removed other properties (mixed buildings, etc.) since I didn't have a lot. I focussed on houses and apartments only. 

<h3>Cleaning</h3>
About the cleaning, I removed rows with empty values when there were not a lot of rows.<br>
I replaced empty values in some cases by the median and by grouping.

<h3>Transforming of categorical columns</h3>
I changed the column "Building condition" from strings to a scale from 6 to 1 (As new to restore).<br>
I used get dummies for the other 2 categorical columns left (district & property type).<br>
I used standardization scaling on the columns. 

<h2>Metrics</h2>
<img src="https://github.com/user-attachments/assets/8df3345d-4336-4084-b7be-454f7e941a4a">
<imgh src="https://github.com/user-attachments/assets/9b8a0b6a-08aa-434c-acdf-9203250440d9">
<br><br>
Time taken for the entire script (from main.py):<br>
<img src="https://github.com/user-attachments/assets/11f7437f-722a-4ec0-8f5d-bdfbe42d4fd3">
<br>
<br>
Testing vs training sets :<br>

I used 20 % of testing vs 80% of training
<br><br>
Here's the link for the presentation bullet points: 
https://docs.google.com/presentation/d/1qRAU-Twhl___lVKYpkEns3eu3bo7ClQg4Tm7zMEEep4/edit?usp=sharing
<br><br>
PS: I was not done with the OOP and cleaning of the files on time. So I added the raw code (which is working but only works if you run the different files one by one) and the unfinished code with OOP (which is not working since the main file is not finished but uploaded anyway). The metrics and efficiency of the model can be checked with the raw code.
In the meantime, I finished cleaning (I didn't change the model, I just cleaned and used OOP and functions only) and everything is here in case you want to run a proper code: https://github.com/therese-debacker/Immo_Eliza_ML_Clean
