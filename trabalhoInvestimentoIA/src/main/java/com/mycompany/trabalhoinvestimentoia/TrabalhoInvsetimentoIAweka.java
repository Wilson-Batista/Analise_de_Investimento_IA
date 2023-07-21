
package com.mycompany.trabalhoinvestimentoia;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class TrabalhoInvsetimentoIAweka {
    public static void main(String[] args) throws Exception {
    DataSource arquivo = new DataSource("/media/wilson/SD_CARD/TCC/Analise_de_Investimento/Variação_Petrobras.csv"
            + "_IA/trabalhoInvestimentoIA/");
    Instances instanciaArquivo = arquivo.getDataSet();
    instanciaArquivo.setClassIndex(instanciaArquivo.numAttributes() - 1);
    
    MultilayerPerceptron mulPer = new MultilayerPerceptron();
    mulPer.setHiddenLayers("10");//neoronios
    mulPer.setLearningRate(0.3);//taxa de aprendizagem
    mulPer.setTrainingTime(1000);//numero de inetraçoes
    
    mulPer.buildClassifier(instanciaArquivo);
    
        for (int i = 0; i < instanciaArquivo.numInstances(); i++) {
            double valorAtual = instanciaArquivo.instance(i).classValue();
            double valorPrevisto = mulPer.classifyInstance(instanciaArquivo.instance(i));
            System.out.println("Ponto " + i + ": Cotação real = " + valorAtual + ", Valor previsto = " + valorPrevisto);
        }
     
    }
}
