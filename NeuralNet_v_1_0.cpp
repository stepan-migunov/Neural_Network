#include "NeuralNet_v_1_0.h"

[[maybe_unused]]
NeuralNet_v_1_0::NeuralNet_v_1_0(const int & input_layer_size)
{
    inputs.resize(input_layer_size);
    qr = new QRandomGenerator(QTime::currentTime().msecsSinceStartOfDay());
}

[[maybe_unused]]
NeuralNet_v_1_0::NeuralNet_v_1_0(const int & input_layer_size, const QVector < int > & layers_sizes)
{
    qr = new QRandomGenerator(QTime::currentTime().msecsSinceStartOfDay());
    inputs.resize(input_layer_size);
    for (int layersSize : layers_sizes)
        this->pushLayer(layersSize);

}

[[maybe_unused]]
NeuralNet_v_1_0::NeuralNet_v_1_0(QTextStream & input_text_stream)
{
    qr = new QRandomGenerator(QTime::currentTime().msecsSinceStartOfDay());
    int input_layer_size;
    input_text_stream >> input_layer_size;
    inputs.list.clear();
    inputs.list.resize(input_layer_size);
    int perceptron_size;
    input_text_stream >> perceptron_size;
    int k;
    for (int i = 0; i < perceptron_size; ++i)
        input_text_stream >> k,
        this->pushLayer(k);

    for (layer & L : perceptron)
        for (neuron & N : L.list)
            for (double & D : N.weights)
                input_text_stream >> D;

}

[[maybe_unused]]
NeuralNet_v_1_0 & NeuralNet_v_1_0::export_perceptron(QTextStream & output_text_stream)
{
    output_text_stream << inputs.size() << " ";
    output_text_stream << perceptron.size() << " ";
    for (const layer & L : perceptron)
        output_text_stream << L.size() << " ";
    for (const layer & L : perceptron)
    {
        output_text_stream << endl;
        for (const neuron & N : L.list)
        {
            output_text_stream << endl;
            for (const double & D : N.weights)
                output_text_stream << D << " ";
        }
    }
    output_text_stream << endl;
    output_text_stream << endl;
    output_text_stream << endl;
    return *this;
}

[[maybe_unused]] [[nodiscard]]
QVector < double > NeuralNet_v_1_0::get_weights_at(int layer_index, int neuron_index) const
{
    return perceptron[layer_index].list[neuron_index].weights;
}

[[maybe_unused]]
NeuralNet_v_1_0 & NeuralNet_v_1_0::set_neuron_weights
        (int layer_index, int neuron_index, const QVector < double > & weights)
{
    if (perceptron[layer_index][neuron_index].weights.size() != weights.size())
        throw std::range_error("size of data QVector<> does not match required");
    perceptron[layer_index][neuron_index].weights = weights;
    return *this;
}

[[maybe_unused]]
NeuralNet_v_1_0 & NeuralNet_v_1_0::pushLayer(int new_layer_size)
{
    layer layer_((QVector < neuron >(new_layer_size)));

    if (perceptron.empty())
        layer_.bind_layer(inputs);
    else
        layer_.bind_layer(perceptron.back());

    perceptron.push_back(layer_);
    for(neuron & n : perceptron.back().list)
        n.qr_=qr;
    for (neuron & N : perceptron.back().list)
        N.setRandom();
    return *this;
}

[[maybe_unused]] [[nodiscard]]
QVector < double > NeuralNet_v_1_0::compute(const QVector < double > & input) const
{

    if (input.size() != inputs.size())
        throw std::range_error("size of data QVector<> does not match required");

    for (int i = 0; i < (inputs.size() < input.size() ? inputs.size() : input.size()); ++i)
        inputs[i].value = input[i];

    for (const layer & L : perceptron)
        for (const neuron & neuron : L.list)
            neuron.evaluate();

    QVector < double > result;
    for (const neuron & i : perceptron.back().list)
        result.push_back(i.value);
    return result;
}

[[maybe_unused]]
NeuralNet_v_1_0 & NeuralNet_v_1_0::backPropagation
        (const QVector < double > & example_input,
         const QVector < double > & example_output,
         double H_Eta = 0.01,
         size_t order = 1
        )
{

    if (example_output.size() != this->perceptron.back().size())
        throw std::range_error("size of example_output QVector<> does not match required");
    if (example_input.size() != this->inputs.size())
        throw std::range_error("size of example_input QVector<> does not match required");
    for (int counter = 0; counter < order; ++counter)
    {
        const auto & exact_answer = this->compute(example_input);
        layer & layer_back = perceptron.back();
        for (int j = 0; j < example_output.size(); ++j)
        {
            const double & v = layer_back[j].value;
            layer_back[j].delta = v * (1 - v) * (v - example_output[j]);
        }
        for (auto l = perceptron.end() - 2; l != perceptron.begin(); l--)
            for (int j = 0; j < l->size(); ++j)
            {
                const auto & v = (*l)[j].value;
                double delta = 0.0;
                for (int k = 0; k < (l + 1)->size(); ++k)
                    delta += (*(l + 1))[k].delta * (*(l + 1))[k][j];
                (*l)[j].delta = delta * v * (1 - v);
            }
        auto layer_front = perceptron.begin();
        for (int j = 0; j < layer_front->size(); ++j)
        {
            double v = (*layer_front)[j].value, d = 0.0;
            for (int k = 0; k < (layer_front + 1)->size(); ++k)
                d += (*(layer_front + 1))[k].delta
                     *
                     (*(layer_front + 1))[k][j];
            (*layer_front)[j].delta = d * v * (1 - v);
        }

        for (layer & L : this->perceptron)
            for (int j = 0; j < L.size(); ++j)
            {
                for (int i = 0; i < L[j].weights.size(); ++i)
                {
                    NeuralNet_v_1_0::neuron & N = L[j];
                    N.weights[i] += -H_Eta * N.delta * N.incoming_signals[i]->value;
                }
            }
    }
    return *this;
}

NeuralNet_v_1_0::layer::layer(const QVector < neuron > & layer_)
{
    this->list.resize(layer_.size());
    for (int i = 0; i < layer_.size(); ++i)
        list[i] = layer_[i];
}

NeuralNet_v_1_0::layer & NeuralNet_v_1_0::layer::bind_layer(const layer & previous_layer)
{
    for (neuron & N : list)
    {
        N.incoming_signals.clear();
        for (const auto & neuron_ : previous_layer.list)
            N.incoming_signals.push_back(neuron_.getPointer());
        N.weights.clear(),
                N.weights.resize(previous_layer.list.size());
    }
    return *this;
}

NeuralNet_v_1_0::neuron * NeuralNet_v_1_0::neuron::getPointer() const
{
    return const_cast<neuron *>(this);
}

NeuralNet_v_1_0::neuron & NeuralNet_v_1_0::neuron::setRandom()
{
    for (double & weight : this->weights)
        weight = qr_->generateDouble();
    return *this;
}

double NeuralNet_v_1_0::neuron::evaluate() const
{
    for (int i = static_cast<int>(this->value = 0); i < incoming_signals.size(); ++i)
        this->value += incoming_signals[i]->value * weights[i];
    this->value = sigma(this->value);
    return this->value;
}

double NeuralNet_v_1_0::neuron::sigma(double val)
{
    return 1 / (1 + exp(-val));
}

double NeuralNet_v_1_0::neuron::operator[](int weight_this_index) const
{
    return this->weights[weight_this_index];
}

NeuralNet_v_1_0::layer & NeuralNet_v_1_0::operator[](int layer_index)
{
    return perceptron[layer_index];
}

NeuralNet_v_1_0::~NeuralNet_v_1_0()
{
    delete qr;
}

NeuralNet_v_1_0::neuron & NeuralNet_v_1_0::layer::operator[](int neuron_index)
{
    return this->list[neuron_index];
}

const NeuralNet_v_1_0::neuron & NeuralNet_v_1_0::layer::operator[](int neuron_index) const
{
    return this->list[neuron_index];
}

NeuralNet_v_1_0::layer & NeuralNet_v_1_0::layer::resize(int new_size)
{
    this->list.resize(new_size);
    return *this;
}

