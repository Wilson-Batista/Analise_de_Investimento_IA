package com.mycompany.analise_de_investimento.ia;
/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 */
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;
/**
 *
 * @author wilson
 */
public class Analise_de_investimentoIA {

    public static void main(String[] args) {
        try {
            // Carregar arquivo CSV de treinamento
            CSVLoader arquivoTreino = new CSVLoader();
            arquivoTreino.setSource(new File("VariaçãoPetrobrasTreino.csv"));
            Instances dadosTreino = arquivoTreino.getDataSet();
            dadosTreino.setClassIndex(dadosTreino.numAttributes() - 1);
            
            // Converter atributo de classe numérico para nominal
            if (dadosTreino.attribute(dadosTreino.numAttributes() - 1).isNumeric()) {
                weka.filters.unsupervised.attribute.NumericToNominal convert = new weka.filters.unsupervised.attribute.NumericToNominal();
                String[] opcao = new String[]{"-R", Integer.toString(dadosTreino.numAttributes())}; // Converter último atributo
                convert.setOptions(opcao);
                convert.setInputFormat(dadosTreino);
                dadosTreino = weka.filters.Filter.useFilter(dadosTreino, convert);
            }
            
            dadosTreino.setClassIndex(dadosTreino.numAttributes() - 1);

            // SVM
            Classifier svm = new SMO();
            svm.buildClassifier(dadosTreino);

            // Carregar arquivo CSV de teste
            CSVLoader arquivoTeste = new CSVLoader();
            arquivoTeste.setSource(new File("VariaçãoPetrobrasTeste.csv"));
            Instances dadosTeste = arquivoTeste.getDataSet();
            dadosTeste.setClassIndex(dadosTeste.numAttributes() - 1);
            
            if (dadosTeste.attribute(dadosTreino.numAttributes() - 1).isNumeric()) {
                weka.filters.unsupervised.attribute.NumericToNominal convert = new weka.filters.unsupervised.attribute.NumericToNominal();
                String[] opcao = new String[]{"-R", Integer.toString(dadosTeste.numAttributes())}; // Converter último atributo
                convert.setOptions(opcao);
                convert.setInputFormat(dadosTeste);
                dadosTeste = weka.filters.Filter.useFilter(dadosTeste, convert);
            }
            
            dadosTreino.setClassIndex(dadosTreino.numAttributes() - 1);

            // Avaliação usando dados de teste
            Evaluation evolucaoTeste = new Evaluation(dadosTreino);
            evolucaoTeste.evaluateModel(svm, dadosTeste);
            System.out.print("\nAvaliação com dados de teste: \n" + evolucaoTeste.toSummaryString());

            // Fazer previsões usando o modelo SVM nos dados de teste
            int acertos = 0;
            int erros = 0;
            int numeroInstancaia = dadosTeste.numInstances();
            for (int i = 0; i < dadosTeste.numInstances(); i++) {
                Instance instancia = dadosTeste.instance(i);
                double realClasse = instancia.classValue();
                double previsaoClasse = svm.classifyInstance(instancia);
                
                if (realClasse == previsaoClasse) {
                    acertos++;
                }else{
                    erros++;
                }
            }
            double taxaDeAcertos = (double) acertos / dadosTeste.numInstances() * 100.0;
            System.out.println("Acertou " + acertos + "\n" + "Taxa de acertos foi de " + taxaDeAcertos + 
                    "%\nNumero de Instancia " + numeroInstancaia);
        } catch (Exception e) {
            System.out.println("erro: " + e);
        }
    }
    
}
