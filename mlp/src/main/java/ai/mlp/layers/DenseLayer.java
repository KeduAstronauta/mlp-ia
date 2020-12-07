package ai.mlp.layers;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import ai.mlp.neurons.Neuron;

public class DenseLayer<E extends Neuron> extends Layer{
	
	public Neuron[] neurons; // Neurônios da camada
	
	public DenseLayer(int n, int x, Class<E> clazz, String label, int idx) {
		super(label, idx);
		try {
			neurons = new Neuron[n];
			for(int i=0; i<n; i++) { // Instancia neurônios
				Constructor<E> constr = clazz.getDeclaredConstructor();
				E ne = constr.newInstance();
				ne.init(x);
				neurons[i] = ne;
			}
		} catch (InstantiationException | IllegalAccessException | NoSuchMethodException | InvocationTargetException e) {
			System.out.println("\nErro ao instanciar o neurônio: " + clazz.getClass().getSimpleName());
			e.printStackTrace();
		}
	}

	@Override
	public double[] run(double x[]) {
		// TODO Auto-generated method stub
		double[] y = new double[neurons.length];	// Saída
		for(int i=0; i<neurons.length; i++) { // Neurônios
			y[i] = neurons[i].activation(x);
		}
		return y;
	}

	@Override
	public void runGradient(double[][] y, double[][] dy, double[] gy) {
		// TODO Auto-generated method stub
		if(super.label.equals("O")) {
			for(int i=0; i<gy.length; i++) {
				neurons[i].gradient = (y[super.idx][i] - dy[super.idx][i]) * neurons[i].g_l(y[super.idx][i]);
			}
		} else if(super.label.equals("H")) {
			DenseLayer<?> next = (DenseLayer<?>) super.next;
			double[] wn = new double[neurons.length];
			for(int i=0; i<neurons.length; i++) {
				for(int j=0; j<next.neurons.length; j++) {
					wn[i] += neurons[j].gradient * next.neurons[j].w[i];
				}
				neurons[i].gradient = wn[i] * neurons[i].g_l(y[super.idx][i]);
			}
		}
	}
	
	public void updateWeights(double[][] gy, double[][] y, double lr) {
		for(int i=0; i<neurons.length; i++) {
			for(int j=0; j<neurons[i].w.length; j++) {
				neurons[i].w[j] = neurons[i].w[j] - lr * gy[super.idx][i] * y[super.idx][i];
			}
		}
	}
}
