import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public class NeuralNetwork {

    private List<SimpleMatrix> weights;
    private List<Double> bias;
    private Function<Double, Double> activation;

    private NeuralNetwork(List<SimpleMatrix> weights, Function<Double, Double> activation, List<Double> bias) {
        this.activation = activation;
        this.weights = weights;
        this.bias = bias;
    }

    public static class Builder {

        private List<SimpleMatrix> weights;
        private Function<Double, Double> activation;
        private List<Double> biases;

        private final Function<Double, Double> sigmoid = x -> {
            return 1 / (1 + Math.exp(-x)); // TODO precalculate for efficiency
        };

        public Builder() {
            weights = new ArrayList<SimpleMatrix>();
            biases = new ArrayList<Double>();
            activation = sigmoid;
        }

        public Builder addLayer(double[][] data, double bias) {
            if (weights.size() == 0 ||
                    weights.get(weights.size() - 1).numCols() == data.length) {
                weights.add(new SimpleMatrix(data));
                biases.add(bias);
            } else {
                throw new IllegalArgumentException("Error: Invalid weights structure"); // TODO a better error message
            } // TODO is it possible to make error messages string resources?
            return this;
        }

        public Builder addLayer(double[][] data) {
            return addLayer(data, 0);
        }

        public Builder

        setActivation(Function<Double, Double> activation) {
            this.activation = activation;
            return this;
        }

        public NeuralNetwork build() {
            return new NeuralNetwork(weights, activation, biases);
        }
    }

    public double[] forwardPropagate(double[] data) {
        SimpleMatrix a = new SimpleMatrix(1, data.length, true, data);
        SimpleMatrix z;

        for (int it = 0; it < weights.size(); it++) {
            z = a.mult(weights.get(it));
            assert z.numRows() == 1;

            for (int c = 0; c < z.numCols(); c++) {
                double activated = activation.apply(z.get(0, c)) + bias.get(it);
                z.set(0, c, activated);
            }
            a = z;
        }

        assert a.getMatrix().data.length == weights.get(weights.size() - 1).numCols();
        return a.getMatrix().data;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Input size: ").append(weights.get(0).numRows()).append("\n");
        sb.append("Output size: ").append(weights.get(weights.size() - 1).numCols()).append("\n");
        for (int i = 0; i < weights.size(); i++) {
            sb.append("Layer ").append(i).append("\n");
            sb.append(weights.get(i).toString());
        }
        return sb.toString();
    }
}
