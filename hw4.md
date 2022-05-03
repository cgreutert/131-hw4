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

Across 10 folds, I will be fitting 30 models to the data. 5.

``` r
log_fit <- log_wkflow %>% fit_resamples(titanic_folds)
lda_fit <- lda_wkflow %>% fit_resamples(titanic_folds)
qda_fit <- qda_wkflow %>% fit_resamples(titanic_folds)
```

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
