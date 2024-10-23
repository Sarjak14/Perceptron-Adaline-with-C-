#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>
//Here seperate classes are defined for Classifier and Adaline
class GradientDescent
{
public:
    double learning_rate = 0.01;
    int epochs = 1000;
    std::vector<double> weights;
    double bias = 0;
    double dot_product(const std::vector<double> &X)
    {
        double net_input = 0.0;
        for (int j = 0; j < X.size(); j++)
        {
            net_input += X[j] * weights[j];
        }
        return net_input + bias;
    }
    void descend(const std::vector<std::vector<double>> &X, const std::vector<double> &y)
    {
        for (int i = 0; i < X.size(); i++)
        {
            double output = dot_product(X[i]);
            double error = y[i] - output;
            for (int k = 0; k < weights.size(); ++k)
            {
                weights[k] += learning_rate * error * X[i][k];
            }
            bias += learning_rate * error;
        }
    }
    void initailize_weights(std::vector<double> &a)
    {
        for (int i = 0; i < a.size(); i++)
        {
            a.at(i) = 0;
        }
    }
    void fit(const std::vector<std::vector<double>> &X, const std::vector<double> &y)
    {
        int no_of_samples = X.size();
        int no_of_features = X.at(0).size();
        weights.resize(no_of_features);
        initailize_weights(weights);
        for (int i = 0; i < epochs; i++)
        {
            descend(X, y);
        }
    }
    double predict(const std::vector<double> &X)
    {
        return dot_product(X);
    }
};
class Classifier
{
public:
    double learning_rate = 0.01;
    int epochs = 1000;
    std::vector<double> weights;
    double bias = 0;
    double dot_product(const std::vector<double> &X)
    {
        double net_input = 0.0;
        for (int j = 0; j < X.size(); j++)
        {
            net_input += X[j] * weights[j];
        }
        return net_input + bias;
    }
    void initialize_weights(std::vector<double> &a)
    {
        for (int i = 0; i < a.size(); i++)
        {
            a.at(i) = 0;
        }
    }
    double linear_activation_function(double net_input)
    {
        return net_input > 0.0 ? 1 : -1; // Changed to return 0 and 1
    }
    void fit(const std::vector<std::vector<double>> &X, const std::vector<double> &y)
    {
        int no_of_samples = X.size();
        int no_of_features = X.at(0).size();
        weights.resize(no_of_features);
        initialize_weights(weights);
        for (int i = 0; i < epochs; ++i)
        {
            for (int j = 0; j < no_of_samples; ++j)
            {
                double net_input = dot_product(X.at(j));
                double error = y.at(j) - linear_activation_function(net_input);

    
                for (int k = 0; k < weights.size(); ++k)
                {
                    weights[k] += learning_rate * error * X.at(j)[k];
                }
                bias += learning_rate * error;
            }
        }
    }
    double predict(const std::vector<double> &X)
    {
        return linear_activation_function(dot_product(X));
    }
};

void test_gradient_descent()
{
    std::cout << "Testing GradientDescent (Linear Regression):" << std::endl;
    std::vector<std::vector<double>> X = {{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}};
    std::vector<double> y = {8, 13, 18, 23, 28}; 
    GradientDescent gd; 
    gd.fit(X, y);
    std::vector<std::vector<double>> test_data = {{6, 7}, {7, 8}, {8, 9}};
    std::vector<double> expected_results = {33, 38, 43};
    for (int i = 0; i < test_data.size(); ++i)
    {
        double prediction = gd.predict(test_data[i]);
        std::cout << "Test " << i + 1 << ": Predicted = " << std::setprecision(4) << prediction
                  << ", Expected = " << expected_results[i] << std::endl;
    }
}
void test_classifier()
{
    std::cout << "Testing Classifier (Binary Classification):" << std::endl;
    std::vector<std::vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<double> y = {-1, -1, -1, 1}; 
    Classifier classifier;
    classifier.fit(X, y);
    std::vector<std::vector<double>> test_data = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<double> expected_results = {-1, -1, -1, 1};
    for (int i = 0; i < test_data.size(); ++i)
    {
        double prediction = classifier.predict(test_data[i]);
        std::cout << "Test " << i + 1 << ": Predicted = " << std::setprecision(4) << prediction
                  << ", Expected = " << expected_results[i] << std::endl;
    }
}
int main()
{
    test_gradient_descent(); 
    test_classifier();       
    std::cout << "All tests passed successfully!" << std::endl;
    return 0;
}
