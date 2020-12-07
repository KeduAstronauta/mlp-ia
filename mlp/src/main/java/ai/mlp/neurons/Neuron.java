package ai.mlp.neurons;

import java.util.Random;

public abstract class Neuron {
	
	public double w[]; 						// Pesos
	public double buffered_w[];				// Pesos atualizados
	public double output;					// Saída do neurônio
	public double gradient;					// Gradient calculado
	protected double bias = 0.0;			// Bias, do qual é alterado pela classe que extende Neuron
	
	abstract public double g(double v);		// Activation Function
	abstract public double g_l(double v);	// Derivative of the Activation Function
	abstract protected void setBias();		// Set the Neuron Bias
	
	public Neuron() {
	}
	
	// Inicialização
	public void init(int s) {
		w = new double[s];
		Random r = new Random();
		for(int i=0; i<w.length; i++) {
			w[i] = r.nextDouble();
		}
		setBias();
	}
	
	// Ativação do neurônio
	public double activation(double x[]) {
		double s = 0;
		for(int i=0; i<x.length; i++) { // Realiza a soma ponderada das entradas com os respectivos pesos
			s += w[i] * x[i];
		}
//		s += w[w.length-1]*bias;		// Adiciona o Bias
		this.output = g(s);
		return this.output;
	}
	
	public void updateWeights() {
		for(int i=0; i<buffered_w.length; i++) {
			w[i] = buffered_w[i];
		}
		buffered_w = null;
	}
}
