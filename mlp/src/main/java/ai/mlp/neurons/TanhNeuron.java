package ai.mlp.neurons;

public class TanhNeuron extends Neuron {
	
	// Neurônio Tangencial Hiperbólica
	public TanhNeuron() {
		super();
		// TODO Auto-generated constructor stub
	}

	@Override
	public double g(double v) {		// Função de ativação
		// TODO Auto-generated method stub
		return Math.tanh(v);
	}

	@Override
	public double g_l(double v) {	// Derivada da Função de ativação
		// TODO Auto-generated method stub
		return 1 - Math.pow(Math.tanh(v), 2);
	}

	@Override
	protected void setBias() {		// Set Bias
		// TODO Auto-generated method stub
		super.bias = -1.0;
	}

}
