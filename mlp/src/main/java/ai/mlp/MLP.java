package ai.mlp;

import java.util.ArrayList;
import java.util.Arrays;

import ai.mlp.layers.DenseLayer;
import ai.mlp.layers.Layer;
import ai.mlp.neurons.LogisticNeuron;
import ai.mlp.neurons.TanhNeuron;

public class MLP {
	
	private ArrayList<Layer> layers; 		// Camadas da rede neural
	private double[][] layers_y;
	
	public MLP() {
		this.layers = new ArrayList<>();
	}
	
	// Rotina de treino da rede neural
	public void train(double x[][], double yd[][], double lr, double e) {
		int epocas = 0;
		
		double em_0 = 0;
		double em_1 = 0;
		double em_epoca = 0;
		
		ArrayList<double[]> em = new ArrayList<>();
		
		do {
			em_0 = em(x, yd); 	// Erro antes dos ajustes
			
			for(int i=0; i<x.length; i++) {
				// Feedfoward
				net_act_training(x[i]);
				
				// Backpropagation
//				double[][] gy = new double[layers.size()][];
//				for(int j=0, v=gy.length-1; j<gy.length; j++, v--) {
//					gy[v] = layers.get(layers.size()-1-j).runGradient(layers_y, yd, (v == gy.length-1 ? null : gy[v+1]));
//				}
				
				// Atualização dos pesos
//				for(int j=0; j<layers.size(); j++) {
//					layers.get(j).updateWeights(gy, layers_y, lr);
//				}
				
			}
			
			this.layers_y = null;
			
			em_1 = em(x, yd); // Erro após ajustes
			
			em_epoca = Math.abs(em_1 - em_0); // Erro da época
			epocas++;
			
			em.add(new double[] {(double) epocas, em_epoca}); // Armazena a época e o erro para plot em gráfico
		} while(em_epoca > e /*epocas < 5000*/);
		
		System.out.println("Treino completo");
		System.out.println("Epocas: " + epocas);
		System.out.println("Erro Última Época: " + em_epoca);
	}
	
	// Ativação da rede
	public double[] net_act(double x[]) {
		double aux[] = x;
		for(int i=0; i<layers.size(); i++) { // Camadas
			aux = layers.get(i).run(aux);
		}
		return aux;
	}

	// Ativação da rede, treino
	private double[] net_act_training(double x[]) {
		double aux[] = x;
		layers_y = new double[layers.size()][];
		for(int i=0; i<layers.size(); i++) { // Camadas
			aux = layers.get(i).run(aux);
			layers_y[i] = aux;
		}
		return aux;
	}
	
	// Erro Quadrático Médio
	public double em(double x[][], double yd[][]) {
		double em = 0;
		for(int i=0; i<x.length; i++) {
			double y[] = net_act(x[i]);
			double s = 0;
			for(int j=0; j<y.length; j++) {
				s += Math.pow(yd[i][j] - y[j], 2);
			}
			em += 0.5*s;
		}
		return (1.0/x.length) * em;
	}
	
	/* TIPOS DE CAMADA DA REDE NEURAL */
	
	// Função Logística
	public void addLogisticLayer(int n, int i, String label) {
		Layer l = new DenseLayer<LogisticNeuron>(n, i, LogisticNeuron.class, label, layers.size());
		if(!layers.isEmpty()) layers.get(layers.size()-1).setNext(l);
		layers.add(l);
	}
	
	// Tangencial Hiperbólica
	public void addTanhLayer(int n, int i, String label) {
		Layer l = new DenseLayer<TanhNeuron>(n, i, TanhNeuron.class, label, layers.size());
		if(!layers.isEmpty()) layers.get(layers.size()-1).setNext(l);
		layers.add(l);
	}
	
	public static void main(String[] args) {
		/* MLP SETUP */
		MLP nn = new MLP();
		nn.addLogisticLayer(2, 3, "H");
		nn.addLogisticLayer(1, 2, "O");
		
		// Entrada
		double x[][] = {
		//		bias	x1	x2
				{1,		1,	1},
				{1,		1,	0},
				{1,		0,	1},
				{1,		0,	0},
		};
		
		// Saída esperada
		double yd[][] = {
				{0},	// 1, 1
				{1},	// 1, 0
				{1},	// 0, 1
				{0},	// 0, 0
		};
		
		double lr = 0.1; // Taxa de aprendizado
		double e = 0.01; // Erro
		
		nn.train(x, yd, lr, e);
		double[] rslt1 = nn.net_act(x[0]);
		System.out.println(Arrays.toString(rslt1));
		double[] rslt2 = nn.net_act(x[1]);
		System.out.println(Arrays.toString(rslt2));
		double[] rslt3 = nn.net_act(x[2]);
		System.out.println(Arrays.toString(rslt3));
		double[] rslt4 = nn.net_act(x[3]);
		System.out.println(Arrays.toString(rslt4));
	}
	
}
