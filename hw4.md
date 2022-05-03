Homework Four: Resampling
================
Carly Greutert

``` r
titanic <- read_csv('C:\\Program Files\\Git\\tmp\\131-hw4\\titanic.csv')
names <- c('pclass', 'survived')
titanic[,names] <- lapply(titanic[,names] , factor)
```

1.  

``` r
set.seed(777)
titanic_split <- initial_split(titanic, prop = 0.80, strata = 'survived')
titanic_train <- training(titanic_split)
titanic_test <- testing(titanic_split)
titanic_recipe <- recipe(survived ~ pclass + sex + age + sib_sp + parch + fare, data = titanic_train)
titanic_recipe <- titanic_recipe %>%
  step_impute_linear(age) %>%
  step_dummy(all_nominal_predictors())%>%
  step_interact(~starts_with("sex"):fare) %>%
  step_interact(~age:fare)
dim(titanic_train)
```

    ## [1] 712  12

``` r
dim(titanic_test)
```

    ## [1] 179  12

From the dim() function above, we can verify the number of observations
was split correctly.  
2.

``` r
titanic_folds <- vfold_cv(titanic_train, v = 10)
titanic_folds
```

    ## #  10-fold cross-validation 
    ## # A tibble: 10 x 2
    ##    splits           id    
    ##    <list>           <chr> 
    ##  1 <split [640/72]> Fold01
    ##  2 <split [640/72]> Fold02
    ##  3 <split [641/71]> Fold03
    ##  4 <split [641/71]> Fold04
    ##  5 <split [641/71]> Fold05
    ##  6 <split [641/71]> Fold06
    ##  7 <split [641/71]> Fold07
    ##  8 <split [641/71]> Fold08
    ##  9 <split [641/71]> Fold09
    ## 10 <split [641/71]> Fold10

3.  What we are doing above is resampling by splitting our training data
    into further subsets in order to provide a better evaluation for our
    model. K-fold cross-validation splits our data K-ways and has K-1
    subsets and one subset for validation. It will help us train our
    model further. We should use k-fold cross-validation, rather than
    simply fitting and testing models on the entire training set in
    order to decide which model produces the lowest error rates and is
    therefore the best model to use. If we did use the entire training
    set, the resampling method used would be the validation set
    approach.  
4.  

``` r
log_reg <- logistic_reg() %>% 
  set_engine("glm") %>% 
  set_mode("classification")
log_wkflow <- workflow() %>% 
  add_model(log_reg) %>% 
  add_recipe(titanic_recipe)
lda_mod <- discrim_linear() %>% 
  set_mode("classification") %>% 
  set_engine("MASS")
lda_wkflow <- workflow() %>% 
  add_model(lda_mod) %>% 
  add_recipe(titanic_recipe)
qda_mod <- discrim_quad() %>% 
  set_mode("classification") %>% 
  set_engine("MASS")
qda_wkflow <- workflow() %>% 
  add_model(qda_mod) %>% 
  add_recipe(titanic_recipe)
```

Across 10 folds, I will be fitting 30 models to the data.  
5.

``` r
log_fit <- log_wkflow %>% fit_resamples(titanic_folds)
log_fit
```

    ## # Resampling results
    ## # 10-fold cross-validation 
    ## # A tibble: 10 x 4
    ##    splits           id     .metrics         .notes          
    ##    <list>           <chr>  <list>           <list>          
    ##  1 <split [640/72]> Fold01 <tibble [2 x 4]> <tibble [0 x 3]>
    ##  2 <split [640/72]> Fold02 <tibble [2 x 4]> <tibble [0 x 3]>
    ##  3 <split [641/71]> Fold03 <tibble [2 x 4]> <tibble [0 x 3]>
    ##  4 <split [641/71]> Fold04 <tibble [2 x 4]> <tibble [0 x 3]>
    ##  5 <split [641/71]> Fold05 <tibble [2 x 4]> <tibble [0 x 3]>
    ##  6 <split [641/71]> Fold06 <tibble [2 x 4]> <tibble [0 x 3]>
    ##  7 <split [641/71]> Fold07 <tibble [2 x 4]> <tibble [0 x 3]>
    ##  8 <split [641/71]> Fold08 <tibble [2 x 4]> <tibble [0 x 3]>
    ##  9 <split [641/71]> Fold09 <tibble [2 x 4]> <tibble [0 x 3]>
    ## 10 <split [641/71]> Fold10 <tibble [2 x 4]> <tibble [0 x 3]>

``` r
lda_fit <- lda_wkflow %>% fit_resamples(titanic_folds)
lda_fit
```

    ## # Resampling results
    ## # 10-fold cross-validation 
    ## # A tibble: 10 x 4
    ##    splits           id     .metrics         .notes          
    ##    <list>           <chr>  <list>           <list>          
    ##  1 <split [640/72]> Fold01 <tibble [2 x 4]> <tibble [0 x 3]>
    ##  2 <split [640/72]> Fold02 <tibble [2 x 4]> <tibble [0 x 3]>
    ##  3 <split [641/71]> Fold03 <tibble [2 x 4]> <tibble [0 x 3]>
    ##  4 <split [641/71]> Fold04 <tibble [2 x 4]> <tibble [0 x 3]>
    ##  5 <split [641/71]> Fold05 <tibble [2 x 4]> <tibble [0 x 3]>
    ##  6 <split [641/71]> Fold06 <tibble [2 x 4]> <tibble [0 x 3]>
    ##  7 <split [641/71]> Fold07 <tibble [2 x 4]> <tibble [0 x 3]>
    ##  8 <split [641/71]> Fold08 <tibble [2 x 4]> <tibble [0 x 3]>
    ##  9 <split [641/71]> Fold09 <tibble [2 x 4]> <tibble [0 x 3]>
    ## 10 <split [641/71]> Fold10 <tibble [2 x 4]> <tibble [0 x 3]>

``` r
qda_fit <- qda_wkflow %>% fit_resamples(titanic_folds)
qda_fit
```

    ## # Resampling results
    ## # 10-fold cross-validation 
    ## # A tibble: 10 x 4
    ##    splits           id     .metrics         .notes          
    ##    <list>           <chr>  <list>           <list>          
    ##  1 <split [640/72]> Fold01 <tibble [2 x 4]> <tibble [0 x 3]>
    ##  2 <split [640/72]> Fold02 <tibble [2 x 4]> <tibble [0 x 3]>
    ##  3 <split [641/71]> Fold03 <tibble [2 x 4]> <tibble [0 x 3]>
    ##  4 <split [641/71]> Fold04 <tibble [2 x 4]> <tibble [0 x 3]>
    ##  5 <split [641/71]> Fold05 <tibble [2 x 4]> <tibble [0 x 3]>
    ##  6 <split [641/71]> Fold06 <tibble [2 x 4]> <tibble [0 x 3]>
    ##  7 <split [641/71]> Fold07 <tibble [2 x 4]> <tibble [0 x 3]>
    ##  8 <split [641/71]> Fold08 <tibble [2 x 4]> <tibble [0 x 3]>
    ##  9 <split [641/71]> Fold09 <tibble [2 x 4]> <tibble [0 x 3]>
    ## 10 <split [641/71]> Fold10 <tibble [2 x 4]> <tibble [0 x 3]>

6.  

``` r
collect_metrics(log_fit)
```

    ## # A tibble: 2 x 6
    ##   .metric  .estimator  mean     n std_err .config             
    ##   <chr>    <chr>      <dbl> <int>   <dbl> <chr>               
    ## 1 accuracy binary     0.801    10  0.0200 Preprocessor1_Model1
    ## 2 roc_auc  binary     0.847    10  0.0206 Preprocessor1_Model1

``` r
collect_metrics(lda_fit)
```

    ## # A tibble: 2 x 6
    ##   .metric  .estimator  mean     n std_err .config             
    ##   <chr>    <chr>      <dbl> <int>   <dbl> <chr>               
    ## 1 accuracy binary     0.795    10  0.0192 Preprocessor1_Model1
    ## 2 roc_auc  binary     0.847    10  0.0211 Preprocessor1_Model1

``` r
collect_metrics(qda_fit)
```

    ## # A tibble: 2 x 6
    ##   .metric  .estimator  mean     n std_err .config             
    ##   <chr>    <chr>      <dbl> <int>   <dbl> <chr>               
    ## 1 accuracy binary     0.801    10  0.0183 Preprocessor1_Model1
    ## 2 roc_auc  binary     0.848    10  0.0188 Preprocessor1_Model1

The means for accuracy and roc_auc all perform similarly across these
three models in terms of the mean. Thus, standard error will be a more
helpful in determining the best fitted model. Above, we see our QDA
fitted model has a 0.018 standard error as opposed to the others 0.019
error, a small but noticeable difference. Therefore, QDA is our best
fitted model since it has the lowest standard error.  
7.

``` r
final_fit <- fit(qda_wkflow, titanic_train)
final_fit
```

    ## == Workflow [trained] ==========================================================
    ## Preprocessor: Recipe
    ## Model: discrim_quad()
    ## 
    ## -- Preprocessor ----------------------------------------------------------------
    ## 4 Recipe Steps
    ## 
    ## * step_impute_linear()
    ## * step_dummy()
    ## * step_interact()
    ## * step_interact()
    ## 
    ## -- Model -----------------------------------------------------------------------
    ## Call:
    ## qda(..y ~ ., data = data)
    ## 
    ## Prior probabilities of groups:
    ##       No      Yes 
    ## 0.616573 0.383427 
    ## 
    ## Group means:
    ##          age    sib_sp     parch     fare pclass_X2 pclass_X3  sex_male
    ## No  29.69526 0.5854214 0.3530752 23.30375 0.1776765 0.6765376 0.8451025
    ## Yes 27.90613 0.4652015 0.4652015 43.95102 0.2783883 0.3663004 0.3260073
    ##     sex_male_x_fare age_x_fare
    ## No         19.58839   709.4941
    ## Yes        11.05737  1379.3999

8.  

``` r
predict(final_fit, new_data = titanic_test, type = "class") %>% 
  bind_cols(titanic_test %>% select(survived)) %>% 
  accuracy(truth = survived, estimate = .pred_class)
```

    ## # A tibble: 1 x 3
    ##   .metric  .estimator .estimate
    ##   <chr>    <chr>          <dbl>
    ## 1 accuracy binary         0.827

My model’s testing accuracy ended up performing very similarly to the
model on the folded data. This is to be expected since the folding of
the data allowed for more training and testing so that it will perform
similarly on new data, which it accomplished with roughly 82% accuracy.
