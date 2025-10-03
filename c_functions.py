from sklearn.linear_model import LinearRegression, LogisticRegression


def linear_regression():
    return """          
    float linear_regression_prediction(double* features, double* thetas, int n_parameters){
        double res = thetas[0];
        for (int i = 0; i < n_parameters - 1; i++){
            res += features[i] * thetas[i+1];
        }
        return res;
    }"""


def pow():
    return """
double pow(double a, double b) {
    float res = 1;
    for (int i = 0; i < b; i++){
        res *= a;
    }
    return res;
}
"""


def factorial():
    return """
float factorial(float a){
    float res = 1;
    while (a > 0){
        res *= a;
        a--;
    }
    return res;
}
"""


def exp_aprox():
    return """
float exp_approx(float x, int n_term) {
    float res = 0;
    for (int i = 0; i <= n_term; i++){
        res += pow(x, i) / factorial(i);
    }
    return res;
}
"""


def sigmoid():
    return """
float sigmoid(float x){
    return 1 / (1 + exp_approx(-x, 10));
}
"""


def list_to_c(python_list: list):
    c_list = "{"
    for i in range(len(python_list)):
        if isinstance(python_list[i], list):
            c_list += list_to_c(python_list[i])
        else:
            c_list += python_list[i]
        if i < len(python_list) - 1:
            c_list += ","
    return c_list + "}"


def logistic_regression():
    return """
float logistic_regression_prediction(double* features, double* thetas, int n_parameters){
    return sigmoid(linear_regression_prediction(features, thetas, n_parameters));
}
"""


def multiclasses_logistic_regression(model: LogisticRegression, features: list):
    return f"""
    void main(){{
        double thetas[{model.coef_.shape[0]}][{model.coef_.shape[1] + 1}] = {list_to_c([[str(bias), *map(str, coefs)] for bias, coefs in zip(model.intercept_, model.coef_)])};
        int n_parameters = {len(model.coef_[0]) + 1};
        char* classes[{len(model.classes_)}] = {list_to_c(list(map(str, model.classes_)))};

        double features[{len(features)}] = {{ {", ".join(features)} }};

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

        printf("Prediction: %s", classes[max_i]);
    }}
    """


def binary_logistic_regression(model: LogisticRegression, features: list):
    return f"""
    void main(){{
        double thetas[{model.coef_.shape[1] + 1}] = {{ {", ".join([str(model.intercept_)] + [str(coef) for coef in model.coef_])} }};
        int n_parameters = {len(model.coef_.shape[1]) + 1};
        char* classes[{len(model.classes_)}] = {list_to_c(list(map(str, model.classes_)))};

        double features[{len(features)}] = {{ {", ".join(features)} }};

        double pred = logistic_regression_prediction(features, thetas, n_parameters);

        int max_i = pred > 0.5 ? 0 : 1;

        printf("Prediction: %s", classes[max_i]);
    }}
    """
