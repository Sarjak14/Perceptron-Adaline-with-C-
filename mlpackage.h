#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <map>

class Measurements
{
public:
    double accuracy_score(const std::vector<int> &y_true, const std::vector<int> &y_pred)
    {
        if (y_true.size() != y_pred.size())
        {
            throw std::invalid_argument("Size of true labels and predicted labels must be the same.");
        }
        int correct_predictions = 0;
        for (size_t i = 0; i < y_true.size(); ++i)
        {
            if (y_true[i] == y_pred[i])
            {
                correct_predictions++;
            }
        }
        return static_cast<double>(correct_predictions) / y_true.size();
    }

    std::map<std::string, int> confusion_matrix(const std::vector<int> &y_true, const std::vector<int> &y_pred)
    {
        if (y_true.size() != y_pred.size())
        {
            throw std::invalid_argument("Size of true labels and predicted labels must be the same.");
        }

        int tp = 0;
        int tn = 0;
        int fp = 0;
        int fn = 0;

        for (size_t i = 0; i < y_true.size(); ++i)
        {
            if (y_true[i] == 1 && y_pred[i] == 1)
            {
                tp++;
            }
            else if (y_true[i] == 0 && y_pred[i] == 0)
            {
                tn++;
            }
            else if (y_true[i] == 0 && y_pred[i] == 1)
            {
                fp++;
            }
            else if (y_true[i] == 1 && y_pred[i] == 0)
            {
                fn++;
            }
        }

        return {{"TP", tp}, {"TN", tn}, {"FP", fp}, {"FN", fn}};
    }
};
class Manipulation
{
public:
    std::vector<std::vector<double>> normalize(const std::vector<std::vector<double>> &X)
    {
        std::vector<std::vector<double>> normalized(X.size(), std::vector<double>(X[0].size()));
        for (size_t j = 0; j < X[0].size(); ++j)
        {
            double min_val = X[0][j];
            double max_val = X[0][j];
            for (size_t i = 0; i < X.size(); ++i)
            {
                if (X[i][j] < min_val)
                    min_val = X[i][j];
                if (X[i][j] > max_val)
                    max_val = X[i][j];
            }
            for (size_t i = 0; i < X.size(); ++i)
            {
                normalized[i][j] = (X[i][j] - min_val) / (max_val - min_val);
            }
        }
        return normalized;
    }

    void details(const std::vector<std::vector<double>> &X)
    {
        for (size_t j = 0; j < X[0].size(); ++j)
        {
            std::vector<double> column;
            for (size_t i = 0; i < X.size(); ++i)
            {
                column.push_back(X[i][j]);
            }
            double mean = std::accumulate(column.begin(), column.end(), 0.0) / column.size();
            std::sort(column.begin(), column.end());
            double median = column.size() % 2 == 0 ? (column[column.size() / 2 - 1] + column[column.size() / 2]) / 2 : column[column.size() / 2];
            double variance = 0.0;
            for (const auto &value : column)
            {
                variance += (value - mean) * (value - mean);
            }
            variance /= column.size();
            double q1 = column[column.size() / 4];
            double q3 = column[3 * column.size() / 4];

            std::cout << "Column " << j + 1 << ":\n";
            std::cout << "Mean: " << mean << "\n";
            std::cout << "Median: " << median << "\n";
            std::cout << "Variance: " << variance << "\n";
            std::cout << "Q1: " << q1 << "\n";
            std::cout << "Q3: " << q3 << "\n";
            std::cout << "---------------------------------\n";
        }
    }
};
class LogisticRegression
{
private:
    int epochs;
    double learning_rate;
    std::vector<double> weights;
    double bias = 0;

    LogisticRegression(int epochs = 1000, double learning_rate = 0.01) : epochs(epochs), learning_rate(learning_rate) {}

    void initialize_weights(int size)
    {
        for (int i = 0; i < size; ++i)
        {
            weights.at(i) = 0;
        }
    }

    double sigmoid_activation_function(double net_input)
    {
        return 1 / (1 + exp(-net_input));
    }

    double net_input(std::vector<double> X)
    {
        double net_input = 0;
        for (int i = 0; i < X.size(); ++i)
        {
            net_input += X.at(i) * weights.at(i);
        }
        return net_input + bias;
    }

    int threshold_function(double probability)
    {
        return probability > 0.5 ? 1 : 0;
    }

    void decend(std::vector<std::vector<double>> X, std::vector<double> y)
    {

        for (int i = 0; i < X.size(); ++i)
        {
            double net_inputs = net_input(X.at(i));
            double prediction = sigmoid_activation_function(net_inputs);
            double error = y.at(i) - prediction;
            for (int j = 0; j < X.at(0).size(); ++j)
            {
                weights.at(j) += learning_rate * error * X.at(i).at(j);
            }
            bias += learning_rate * error;
        }
    }

public:
    void fit(std::vector<std::vector<double>> X, std::vector<double> y)
    {
        int feature_size = X.at(0).size();
        weights.resize(feature_size);
        initialize_weights(feature_size);
        for (int i = 0; i < epochs; ++i)
        {
            decend(X, y);
        }
    }
    int predict(std::vector<double> X)
    {
        return threshold_function(sigmoid_activation_function(net_input(X)));
    }
};
class GradientDescent
{
private:
    double learning_rate;
    int epochs;
    std::vector<double> weights;
    double bias = 0;
    GradientDescent(int epochs = 1000, double learning_rate = 0.01) : epochs(epochs), learning_rate(learning_rate) {}
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

public:
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
private:
    double learning_rate;
    int epochs;
    std::vector<double> weights;
    double bias = 0;
    Classifier(int epochs = 1000, double learning_rate = 0.01) : epochs(epochs), learning_rate(learning_rate) {}
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
        return net_input > 0.0 ? 1 : -1;
    }

public:
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
