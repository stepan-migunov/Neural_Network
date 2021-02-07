//
// Created by stepan on 18.01.2021.
//

#ifndef QT_PROBE_FIRST_NEURALNET_V_1_0_H
#define QT_PROBE_FIRST_NEURALNET_V_1_0_H

#include <qexception.h>
#include <QTextStream>
#include <QRandomGenerator>
#include <QTime>
#include <QVector>


class NeuralNet_v_1_0
{
private:
    struct neuron
    {
    private:
        QVector< neuron * >    incoming_signals;
        QVector< double >      weights;
        mutable double value;
        static double sigma(double val) ;
        friend class NeuralNet_v_1_0;
        double delta;
        QRandomGenerator* qr_;
    public:
        [[nodiscard]] neuron *getPointer() const;
        double evaluate() const;
        [[maybe_unused]]neuron & setRandom();
        [[nodiscard]][[maybe_unused]] double operator[](int weight_this_index)const;
    };

    struct layer
    {
        friend class NeuralNet_v_1_0;
    private:
        layer & resize(int new_size);

    public:
        QVector
        <neuron> list;
        [[nodiscard]] size_t size() const
        {
            return list.size();
        }
        layer() = default;
        explicit layer(const QVector
        <neuron> &layer_);
        layer & bind_layer(const layer &previous_layer);
                 neuron & operator[](int neuron_index);
        const    neuron & operator[](int neuron_index) const;
    };

    QVector<layer> perceptron;
    mutable layer inputs;
    QRandomGenerator* qr;
public:
    [[maybe_unused]] explicit   NeuralNet_v_1_0(const int &input_layer_size);
    [[maybe_unused]] explicit   NeuralNet_v_1_0(const int &input_layer_size, const QVector
    <int> &layers_sizes);
    [[maybe_unused]] explicit   NeuralNet_v_1_0(QTextStream& input_text_stream);
    ~NeuralNet_v_1_0();

    [[maybe_unused]] NeuralNet_v_1_0 & export_perceptron(QTextStream& output_text_stream);
    [[maybe_unused]] layer &operator[](int layer_index);

    [[maybe_unused]] NeuralNet_v_1_0 & pushLayer
    (int new_layer_size);

    [[maybe_unused]] [[nodiscard]] QVector<double> compute
    (const QVector
    <double> &input) const;

    [[maybe_unused]] NeuralNet_v_1_0 &set_neuron_weights
    (int layer_index, int neuron_index, const QVector
    <double>& weights);

    [[maybe_unused]] [[nodiscard]] QVector<double> get_weights_at(int layer_index, int neuron_index) const;

    //TODO: void backPropagation
    [[maybe_unused]]  NeuralNet_v_1_0 & backPropagation
            (const QVector < double > & example_input,
             const QVector < double > & example_output, double H_Eta,
             size_t order
            );
};


#endif //QT_PROBE_FIRST_NEURALNET_V_1_0_H
