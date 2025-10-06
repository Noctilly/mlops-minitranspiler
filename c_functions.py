from sklearn.linear_model import LinearRegression, LogisticRegression


def linear_regression():
    return """          
    double linear_regression_prediction(double* features, double* thetas, int n_parameters){
        double res = thetas[0];
        for (int i = 0; i < n_parameters - 1; i++){
            res += features[i] * thetas[i+1];
        }
        return res;
    }"""


def pow():
    return """
double pow(double a, int b) {
    double res = 1;
    for (int i = 0; i < b; i++){
        res *= a;
    }
    return res;
}
"""


def factorial():
    return """
double factorial(int a){
    double res = 1;
    while (a > 0){
        res *= a;
        a--;
    }
    return res;
}
"""


def exp_aprox():
    return """
double exp_approx(double x, int n_term) {
    if (x < 0)
    {
        return 1.0 / exp_approx(-x, n_term);
    }

    double res = 0;
    for (int i = 0; i <= n_term; i++){
        res += pow(x, i) / factorial(i);
    }
    return res;
}
"""


def sigmoid():
    return """
double sigmoid(double x){
    return 1 / (1 + exp_approx(-x, 10));
}
"""


def list_to_c(python_list: list):
    c_list = "{"
    for i in range(len(python_list)):
        if isinstance(python_list[i], list):
            c_list += list_to_c(python_list[i])
        else:
            if isinstance(python_list[i], str):
                c_list += f'"{python_list[i]}"'
            else:
                c_list += str(python_list[i])
        if i < len(python_list) - 1:
            c_list += ","
    return c_list + "}"


def logistic_regression():
    return """
double logistic_regression_prediction(double* features, double* thetas, int n_parameters){
    return sigmoid(linear_regression_prediction(features, thetas, n_parameters));
}
"""


def multiclasses_logistic_regression_main(model: LogisticRegression, features: list):
    return f"""
void main(){{
    double thetas[{model.coef_.shape[0]}][{model.coef_.shape[1] + 1}] = {list_to_c([[bias, *coefs] for bias, coefs in zip(model.intercept_, model.coef_)])};
    int n_parameters = {len(model.coef_[0]) + 1};
    char* classes[{len(model.classes_)}] = {list_to_c(list(map(str, model.classes_)))};

    double features[{len(features)}] = {{ {", ".join(map(str, features))} }};

    int max_i = 0;
    double max = -1;

    for (int i = 0; i < {model.coef_.shape[0]}; i++)
    {{
        double pred = logistic_regression_prediction(features, thetas[i], n_parameters);
        if (pred > max)
        {{
            max = pred;
            max_i = i;
        }}
    }}

    printf("Predicted class: %s\\n", classes[max_i]);
}}
    """


def binary_logistic_regression_main(model: LogisticRegression, features: list):
    return f"""
void main(){{
    double thetas[{model.coef_.shape[1] + 1}] = {list_to_c(model.intercept_ + model.coef_[0])};
    int n_parameters = {model.coef_.shape[1] + 1};
    char* classes[{len(model.classes_)}] = {list_to_c(list(map(str, model.classes_)))};

    double features[{len(features)}] = {{ {", ".join(map(str, features))} }};

    double pred = logistic_regression_prediction(features, thetas, n_parameters);

    int max_i = pred < 0.5 ? 0 : 1;

    printf("Predicted class: %s\\n", classes[max_i]);
}}
    """


def linear_regression_main(model: LinearRegression, features: list):
    return f"""    
void main(){{
    double thetas[{len(model.coef_) + 1}] = {{ {", ".join([str(model.intercept_)] + [str(coef) for coef in model.coef_])} }};
    int n_parameters = {len(model.coef_) + 1};

    double features[{len(features)}] = {{ {", ".join(map(str, features))} }};

    printf("Prediction: %f\\n", linear_regression_prediction(features, thetas, n_parameters));
    }}"""
