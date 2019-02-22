import org.junit.Before;
import org.junit.Test;

import java.util.function.Function;

import static org.junit.Assert.assertEquals;

public class NeuralNetworkTests {

    private NeuralNetwork n;
    private NeuralNetwork n2;

    private double[][] weightsOne = {
            {.5, .23, .6},
            {.6, .1, .4},
            {.4, .5, .8}
    };

    private double[][] weightsTwo = {
            {.2, .7},
            {.9, .3},
            {.6, .86}
    };

    private double[][] weightsThree = {
            {1, 1},
            {1, 1}
    };

    private double[][] weightsFour = {
            {1},
            {1}
    };

    private Function<Double, Double> sigmoid = new Function<Double, Double>() {
        @Override
        public Double apply(Double x) {
            return 1 / (1 + Math.exp(-x)); // TODO precalculate for efficiency
        }
    };

    private Function<Double, Double> doNothing = new Function<Double, Double>() {
        @Override
        public Double apply(Double x) {
            return x;
        }
    };

    @Before
    public void init() {
        n = new NeuralNetwork.Builder()
                .addLayer(weightsOne, .5)
                .addLayer(weightsTwo, 0)
                .setActivation(sigmoid)
                .build();

        n2 = new NeuralNetwork.Builder()
                .addLayer(weightsThree, 0)
                .addLayer(weightsFour, .25)
                .setActivation(doNothing)
                .build();
    }

    @Test
    public void testForwardProp() {
        double[] x = {.2, .8, .3};
        double[] y = n.forwardPropagate(x);
        assertEquals(y.length, 2);
    }

    @Test
    public void testForwardPropResult() {
        double[] x = {1, 1};
        double[] y = n2.forwardPropagate(x);
        double[] expected = {4.25};
        assertEquals(expected.length, y.length);
        assertEquals(expected[0], y[0], .001);
    }

    @Test
    public void testPrintNetwork() {
        String expected = "Input size: 3\n" +
                "Output size: 2\n" +
                "Layer 0\n" +
                "Type = dense real , numRows = 3 , numCols = 3\n" +
                " 0.500   0.230   0.600  \n" +
                " 0.600   0.100   0.400  \n" +
                " 0.400   0.500   0.800  \n" +
                "Layer 1\n" +
                "Type = dense real , numRows = 3 , numCols = 2\n" +
                " 0.200   0.700  \n" +
                " 0.900   0.300  \n" +
                " 0.600   0.860  \n";
        assertEquals(expected, n.toString());
    }

    @Test (expected = IllegalArgumentException.class)
    public void testIllegalLayer(){

        NeuralNetwork n = new NeuralNetwork.Builder()
                .addLayer(weightsOne)
                .addLayer(weightsThree)
                .build();
    }
}
