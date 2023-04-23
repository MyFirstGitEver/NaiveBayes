package org.example;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.simple.Sentence;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;

import java.io.*;
import java.util.*;
import java.util.function.Consumer;
import java.util.regex.MatchResult;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

class TextProcessing{
    static class FrequencyTable{
        public FrequencyTable(int positiveOccurrence, int negativeOccurrence) {
            this.positiveOccurrence = positiveOccurrence;
            this.negativeOccurrence = negativeOccurrence;
        }

        double positiveOccurrence;
        double negativeOccurrence;

        void calculatePercent(int dPositive, int dNegative){
            // word order
            // su lien quan giua cac tu
            // P(tweet | pos) = tich(P(word | pos))
            positiveOccurrence /= dPositive;
            negativeOccurrence /= dNegative;
        }
    }

    static List<String> lemmas(String doc){
        Properties props = new Properties();
        props.put("annotators", "tokenize, ssplit, pos, lemma");

        StanfordCoreNLP pipeline;
        pipeline = new StanfordCoreNLP(props, false);
        Annotation document = pipeline.process(doc);

        List<String> lemmas = new ArrayList<>();
        for(CoreMap sentence: document.get(CoreAnnotations.SentencesAnnotation.class))
        {
            for(CoreLabel token: sentence.get(CoreAnnotations.TokensAnnotation.class))
            {
                String lemma = token.get(CoreAnnotations.LemmaAnnotation.class);

                lemmas.add(lemma.toLowerCase());
            }
        }

        return lemmas;
    }

    static void removeStopWordsAndWeirdStrings(List<String> words) throws IOException {
        Set<String> removed = new TreeSet<>();

        for(String word : words){
//            if(word.equals(":)") || word.equals(":-)") || word.equals(":(") || word.equals(":-(")){
//                continue;
//            }

            if(Pattern.compile("^.*[^a-zA-Z].*$").matcher(word).find()){
                removed.add(word);
            }
        }

        BufferedReader reader = new BufferedReader(new FileReader("english"));
        String line;

        while((line = reader.readLine()) != null) {
            if(words.contains(line)){
                removed.add(line);
            }
        }

        for(int i=words.size() - 1;i>=0;i--){
            if(removed.contains(words.get(i))) {
                words.remove(i);
            }
        }
    }
}

class NaiveBayes {
    private final HashMap<String, TextProcessing.FrequencyTable> table;
    private final int positiveOccurrences, negativeOccurrences;
    private int wordsInPositive, wordsInNegative, positiveCount, negativeCount;

    static float checkPos = 0;
    static float checkNeg = 0;
    NaiveBayes(List<Pair<List<String>, Integer>> dataset, int positiveCount, int negativeCount) {
        HashMap<String, TextProcessing.FrequencyTable> table = new HashMap<>();

        int positiveOccurrences = 0, negativeOccurrences = 0;
        for(Pair<List<String>, Integer> pair : dataset) {
            for(String word : pair.first) {
                if(!table.containsKey(word)) {
                    if(pair.second == 0){
                        wordsInNegative++;
                    }
                    else{
                        wordsInPositive++;
                    }

                    table.put(word, new TextProcessing.FrequencyTable(0, 0));
                }
                else{
                    TextProcessing.FrequencyTable t = table.get(word);

                    if(pair.second == 0) {
                        t.negativeOccurrence++;
                        negativeOccurrences++;
                    }
                    else{
                        t.positiveOccurrence++;
                        positiveOccurrences++;
                    }
                }
            }
        }

        double posCheck = 0;
        for(Map.Entry<String, TextProcessing.FrequencyTable> entry : table.entrySet()){
            posCheck += (entry.getValue().positiveOccurrence + 1.0f) / (wordsInPositive + positiveOccurrences);
        }

        this.table = table;
        this.negativeOccurrences = negativeOccurrences;
        this.positiveOccurrences = positiveOccurrences;

        this.positiveCount = positiveCount;
        this.negativeCount = negativeCount;
    }

    public double predict(String tweet) throws IOException {
        List<String> words = TextProcessing.lemmas(tweet);
        TextProcessing.removeStopWordsAndWeirdStrings(words);

        Set<String> set = new TreeSet<>(words);

        double pred = 1;

        for(String word : set){
            pred *= lambda(word);
        }

        return (positiveCount / (float)negativeCount) * pred;
    }

    private double lambda(String word) {
        TextProcessing.FrequencyTable freq = table.get(word);

        double posPercent, negPercent;
        if(freq == null){
            return 1;
        }

        posPercent = (freq.positiveOccurrence + 1.0f) / (wordsInPositive + positiveOccurrences);
        negPercent = (freq.negativeOccurrence + 1.0f) / (wordsInNegative + negativeOccurrences);

        return posPercent / negPercent;
    }
}

public class Main {
    static List<Pair<List<String>, Integer>> dataset;
    static int positiveCount, negativeCount;
    public static void main(String[] args) throws IOException {
        dataset = new ArrayList<>();

        readData("D:\\Source code\\Java\\data\\train pos.txt", 1);
        readData("D:\\Source code\\Java\\data\\train neg.txt", 0);

        NaiveBayes model = new NaiveBayes(dataset, positiveCount, negativeCount);

        FileInputStream fIn = new FileInputStream("D:\\Source code\\Java\\data\\test pos.txt");
        String[] data = new String(fIn.readAllBytes()).split("\3");
        fIn.close();

        int total = data.length;
        int hit = 0;

        for(String tweet : data) {
            double pred = model.predict(tweet);

            boolean isPositive = false;
            if(pred >= 1.0f){
                isPositive = true;
            }

            if(isPositive){
                hit++;
            }
        }

        fIn = new FileInputStream("D:\\Source code\\Java\\data\\test neg.txt");
        data = new String(fIn.readAllBytes()).split("\3");
        fIn.close();

        total += data.length;

        for(String tweet : data) {
            double pred = model.predict(tweet);

            boolean isPositive = false;
            if(pred >= 1.0f){
                isPositive = true;
            }

            if(!isPositive){
                hit++;
            }
        }

        System.out.println(hit / (float)total * 100 + "%");

    }

    static void readData(String fileName, int label) throws IOException {
        FileInputStream fIn = new FileInputStream(fileName);
        String[] data = new String(fIn.readAllBytes()).split("\3");

        for(String tweet : data){
            List<String> words = TextProcessing.lemmas(tweet);
            TextProcessing.removeStopWordsAndWeirdStrings(words);

            dataset.add(new Pair<>(words, label));
        }

        fIn.close();

        if(label == 0){
            negativeCount = data.length;
        }
        else{
            positiveCount = data.length;
        }
    }
}