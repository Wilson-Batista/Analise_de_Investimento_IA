/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 */

package com.mycompany.analise_de_investimento.ia;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import java.io.File;
/**
 *
 * @author wilson
 */
public class Analise_de_investimentoIA {

    public static void main(String[] args) {
        try {
            //Arquivo para treinamento
            CSVLoader arquivo = new CSVLoader();
            arquivo.setSource(new File("VariaçãoPetrobrasTreino.csv"));
            Instances instaciaTreino = arquivo.getDataSet();
            System.out.println("arquivo " + instaciaTreino);
            //
            
        } catch (Exception e) {
            System.out.println("erro: " + e);
        }
    }
}
