Model : Linear regression
Features used :
- property type : house, villa, triplex, ...
- Living area : the stringest correlation with the price
- surface of the plot
- Building condiiton : changed to a scale from 6 to 1 (As new to restore)
- Swimming pool
- District : broader than the zip code but smaller than the province
- Mean income in the zip code (statbel)
- Median price of apartment or house (statbel)

I used the features with the most correlation from the immoweb dataset except when there was a too strong correlation with another parameter (like bedrooms and living area)
I found the added features on statbel.
I tested a lot of other parameters. Some had also a good correlation with the price like number of transactions (apartments and houses sold), population per surface per district, number of parcels built per district 
But these features were too correlated with the mean income or median price. 
I tested also other parameters but the correlation were not good.
I scraped immoweb again to get other information : construction year, parking, ... but the corrlation was not good. So I used the previous dataset I had since it was bigger.

I removed houses more expensive than 2 500 000 et apartments more expensive than 1 000 000 --> I got a better result with the model.
I also removed othe properties (mixed buildings, etc.) since I didn't have a lot. I focussed on houses and apartments. 

Metrics:
![image](https://github.com/user-attachments/assets/557bceed-52ae-4a23-bc23-57376fe81cad)

Testing vs training sets :
I used 20 % of testing vs 80% of training

PS : I'm not finished with the cleaned code so it's not working. I added the uncleaned files so the efficiency of the model and metrics can be checked.
