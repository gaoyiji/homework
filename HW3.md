AUTHOR: Jyun-Yu Cheng

Yiji Gao

# Question 3 : Predictive model building: green certification

Our goal for this problem: find the best predictive model. The revenue
per square foot per calendar year and use this model to quantify the
average change in rental income per square foot(whether in absolute or
percentage terms) associated with green certification.

Step description Step1. Build many models Step2. Model selection(compare
the RMSE of every model and choose the best model) Step3. write the
report tells that why you choose this method, modeling choice and
conclusion

**Step 1 : Build the model** # Step 1-1: Seperate the dataset into the
training-set and testing-set

Now we get the best linear model is : revenue ~ size + empl\_gr +
stories + age + renovated + class\_a + class\_b + green\_rating + net +
amenities + total\_dd\_07 + Precipitation + Gas\_Costs +
Electricity\_Costs + City\_Market\_Rent + size:City\_Market\_Rent +
stories:class\_a + empl\_gr:Electricity\_Costs + size:Precipitation +
green\_rating:amenities + age:total\_dd\_07 + age:green\_rating +
class\_a:Electricity\_Costs + amenities:Electricity\_Costs +
class\_a:Gas\_Costs + age:class\_a + empl\_gr:Precipitation +
class\_a:Precipitation + age:City\_Market\_Rent + class\_a:total\_dd\_07
+ stories:total\_dd\_07 + renovated:City\_Market\_Rent +
renovated:total\_dd\_07 + Electricity\_Costs:City\_Market\_Rent +
stories:renovated + size:renovated + size:age + size:stories +
total\_dd\_07:Precipitation + Precipitation:Electricity\_Costs +
size:Electricity\_Costs + net:City\_Market\_Rent + class\_a:amenities +
total\_dd\_07:Gas\_Costs + Gas\_Costs:Electricity\_Costs +
empl\_gr:Gas\_Costs + amenities:Gas\_Costs + amenities:Precipitation +
class\_a:City\_Market\_Rent + class\_b:Precipitation

![](HW3_files/figure-markdown_strict/unnamed-chunk-5-1.png)

![](HW3_files/figure-markdown_strict/unnamed-chunk-6-1.png)

    ## [1] 1.0444

![](HW3_files/figure-markdown_strict/unnamed-chunk-7-1.png)

    ## Distribution not specified, assuming gaussian ...

    ## [1] 9.512222

![](HW3_files/figure-markdown_strict/unnamed-chunk-8-1.png)

**Step 2 : Compare their RMSE to find out which one is the best
predictive model**

    ## [1] 10.74185

    ## [1] 7.555973

    ## [1] 10.7056

    ## [1] 1.0444

    ## [1] 9.512222

We compare all model’s RMSE , and it shows that Bagging model has the
lowest rmse, the second best model us Random Forest model.

Now we choose the 2 smallest RMSE model to calculate the k-fold
cross-validation standard error

the result shows that Random\_forest\_model with mtry=9 is the best
prediction model

Then we try to know the importance of each variable
![](HW3_files/figure-markdown_strict/unnamed-chunk-11-1.png)![](HW3_files/figure-markdown_strict/unnamed-chunk-11-2.png)

Because the random forest model is not a linear model, it’s hard to
measure the partial effect of green certification. As a result, we
decide to use the average partial of green rating to know more precise
effect of green certification on revenue.

We calculate the difference of predicted revenue on “green
certified”(dummy variable), then take the average of the difference We
can see that the average effect of green certification on the rent
income is almost 0.

    ## [1] 0

**Step 3 : Conclusion**

The best predictive models for revenue is the Random Forest
Model.Holding all else fixed, the average change in rental income per
square foot related to green certification, is almost 0 dollars per
square foot.

# Question 4: Predictive Model Building: California Housing

![](HW3_files/figure-markdown_strict/unnamed-chunk-15-1.png)

    ## [1] 52335.49

![](HW3_files/figure-markdown_strict/unnamed-chunk-15-2.png)

    ## Distribution not specified, assuming gaussian ...

Calculate the RMES of each model

    ## [1] 52335.49

    ## [1] 50924.51

    ## [1] 84479.33

    ## [1] 54220.52

We decide to choose model\_bagging as the best predictive model because
it has the smallest RMSE.

*Plot the pictures * The plot of the original data
![](HW3_files/figure-markdown_strict/unnamed-chunk-20-1.png) We can see
that in the California, the high actual median house value usually
located in the middle and south of the western coast of California.

The plot of model’s prediction of median House value
![](HW3_files/figure-markdown_strict/unnamed-chunk-21-1.png) From the
above figure, we can see that the distribution of predicted value are
very simliar to the real values.

The plot of model’s errors/residuals
![](HW3_files/figure-markdown_strict/unnamed-chunk-22-1.png) The
absolute values of errors are really low, so we can say that this is a
good model to predict

**Conclusion** Our predictive model works well, we can see that high
median value really located in the middle and south of the Western coast
of California.
