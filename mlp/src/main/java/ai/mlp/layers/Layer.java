package ai.mlp.layers;

public abstract class Layer {
	
	public String label;
	public int idx;
	protected Layer next;

	// Executa a função da camada
	public abstract double[] run(double[] x);
	public abstract void runGradient(double[][] y, double[][] dy, double[] gy);
	public abstract void updateWeights(double[][] gy, double[][] y, double lr);
	
	public Layer(String label, int idx) {
		this.label = label;
		this.idx = idx;
		this.next = null;
	}
	
	public void setNext(Layer l) { this.next = l; }
}
