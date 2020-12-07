package ai.mlp.neurons;

public class LogisticNeuron extends Neuron {
	
	// Neurônio Função Logística
	public LogisticNeuron() {
		super();
		// TODO Auto-generated constructor stub
	}

	@Override
	public double g(double v) {		// Função de ativação
		// TODO Auto-generated method stub
		return 1/(1+Math.pow(Math.E, -v));
	}

	@Override
	public double g_l(double v) {	// Derivada da Função de ativação
		// TODO Auto-generated method stub
		return this.g(v)*this.g(-v);
	}

	@Override
	protected void setBias() {		// Set Bias
		// TODO Auto-generated method stub
		super.bias = 1.0;
	}

}
