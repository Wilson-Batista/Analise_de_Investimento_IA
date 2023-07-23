/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.mycompany.analise_de_investimento.ia;

import java.io.File;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.classifiers.functions.MultilayerPerceptron;

/**
 *
 * @author wilson
 */
public class redeNeural {

    public static void main(String[] args) {
        try {
            //Arquivo para treinamento
            CSVLoader arquivoTreino = new CSVLoader();
            arquivoTreino.setSource(new File("VariaçãoPetrobrasTreino.csv"));
            Instances dadosTreino = arquivoTreino.getDataSet();
            dadosTreino.setClassIndex(dadosTreino.numAttributes() - 1);

            //Treino
            MultilayerPerceptron redeNeural = new MultilayerPerceptron();
            redeNeural.setHiddenLayers("10"); // Uma camada oculta com 5 neurônios
            redeNeural.setLearningRate(0.1); // Taxa de aprendizado de 0.1
            redeNeural.setMomentum(0.2); // Momentum de 0.2
            redeNeural.setTrainingTime(2000); // Número máximo de iterações (épocas) = 2000
            redeNeural.setSeed(123); // Semente aleatória para reprodutibilidade
            redeNeural.buildClassifier(dadosTreino);

            //Arquivo para teste
            CSVLoader arquivoTeste = new CSVLoader();
            arquivoTeste.setSource(new File("VariaçãoPetrobrasTeste.csv"));
            Instances dadosTeste = arquivoTeste.getDataSet();
            dadosTeste.setClassIndex(dadosTeste.numAttributes() - 1);
            
            Evaluation previsao = new Evaluation(dadosTeste);
            previsao.evaluateModel(redeNeural, dadosTeste);
            System.out.println(previsao.toSummaryString("\nResultados da Avaliação\n      \n", false));
            
            //Previsão
            int numeroInstancaia = dadosTeste.numInstances();
            int acertos = 0;
            for (int i = 0; i < numeroInstancaia; i++) {
                double valorPrevisto = redeNeural.classifyInstance(dadosTeste.instance(i));
                double valorAtual = dadosTeste.instance(i).classValue();
                
                if (valorPrevisto == valorAtual) {
                    acertos++;
                }
            }
            double taxaDeAcertos = (double) acertos / dadosTeste.numInstances() * 100.0;
            System.out.println("Acertou " + acertos + "\n" + "Taxa de acertos foi de " + taxaDeAcertos + 
                    "%\nNumero de Instancia " + numeroInstancaia);
        } catch (Exception e) {
            System.out.println("Erro: " + e);
        }
    }
}
