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
import java.util.logging.Level;
import java.util.logging.Logger;
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
    }

    static List<String> lemmas(String doc){
        PrintStream err = System.err;
        System.setErr(new PrintStream(new OutputStream() {
            public void write(int b) {
            }
        }));

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

        System.setErr(err);
        return lemmas;
    }

    static void removeStopWordsAndWeirdStrings(Collection<String> words,
                                               boolean removeStopwords,
                                               boolean removeWeird, boolean removeHashTag) throws IOException {
        Set<String> removed = new TreeSet<>();

        if(removeWeird) {
            for(String word : words) {
                if(word.length() > 1 && word.charAt(0) == '#') {
                    removed.add(word);

                    if(!removeHashTag) {
                        removed.add(word.substring(1));
                    }
                }

                if(!Pattern.compile("^[a-zA-Z]+").matcher(word).find()){
                    removed.add(word);
                }
            }
        }

        if(removeStopwords) {
            BufferedReader reader = new BufferedReader(new FileReader("english"));
            String line;

            while((line = reader.readLine()) != null) {
                if(words.contains(line)){
                    removed.add(line);
                }
            }
        }

        for(String word : removed) {
            words.removeIf(word::equals);
        }
    }
}

class NaiveBayes {
    private final HashMap<String, TextProcessing.FrequencyTable> table;
    private final int positiveOccurrences, negativeOccurrences, positiveCount, negativeCount;
    private int wordsInPositive, wordsInNegative;

    NaiveBayes(List<Pair<List<String>, Integer>> dataset, int positiveCount, int negativeCount) {
        HashMap<String, TextProcessing.FrequencyTable> table = new HashMap<>();

        int positiveOccurrences = 0, negativeOccurrences = 0;
        int timer = 0;
        for(Pair<List<String>, Integer> pair : dataset) {
            for(String word : pair.first) {
                if(!table.containsKey(word)) {
                    if(pair.second == 0) {
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
                    }
                    else{
                        t.positiveOccurrence++;
                    }
                }
            }

            if(pair.second == 0){
                positiveOccurrences += pair.first.size();
            }
            else{
                negativeOccurrences += pair.first.size();
            }

            timer++;

            if(timer % 100 == 0) {
                System.out.println(timer + " / " + dataset.size() + " documents processed!");
            }
        }

        this.table = table;
        this.negativeOccurrences = negativeOccurrences;
        this.positiveOccurrences = positiveOccurrences;

        this.positiveCount = positiveCount;
        this.negativeCount = negativeCount;
    }

    public double predict(String tweet) throws IOException {
        List<String> words = TextProcessing.lemmas(tweet);
        TextProcessing.removeStopWordsAndWeirdStrings(words, true, true, false);

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
    static String path = "D:\\Source code\\Outer data\\tweets-for-naive-bayes\\";
    static String path2 = "D:\\Source code\\Outer data\\BOW\\word2vec-nlp-tutorial\\labeledTrainData\\";
    static List<Pair<List<String>, Integer>> dataset;
    static int positiveCount, negativeCount;
    public static void main(String[] args) throws IOException {
        dataset = new ArrayList<>();

        readData(path2 + " 10_000 reviews.txt",
                -1, true);

        NaiveBayes model = new NaiveBayes(dataset, positiveCount, negativeCount);

        FileInputStream fIn = new FileInputStream(path2 + "5_000 tests.txt");
        String[] data = new String(fIn.readAllBytes()).split("\3");
        fIn.close();

        int total = data.length;
        int hit = 0, timer = 0;

        for(String tweet : data) {
            String[] info = tweet.split("\2");
            double pred = model.predict(info[0]);
            boolean isPositive = pred >= 1.0f;

            if(isPositive && info[1].equals("1")) {
                hit++;
            }
            else if(!isPositive && info[1].equals("0")){
                hit++;
            }

            timer++;

            if(timer % 100 == 0) {
                System.out.println("Tested on " + timer + " / " + data.length + " documents processed. Hit: " + hit);
            }
        }

        System.out.println(hit / (float)total * 100 + "%");
    }

    static void firstDataset() throws IOException {
        dataset = new ArrayList<>();

        readData(path + "train pos.txt", 1, false);
        readData(path + "train neg.txt", 0, false);

        NaiveBayes model = new NaiveBayes(dataset, positiveCount, negativeCount);

        FileInputStream fIn = new FileInputStream(path + "test pos.txt");
        String[] data = new String(fIn.readAllBytes()).split("\3");
        fIn.close();

        int total = data.length;
        int hit = 0;

        for(String tweet : data) {
            double pred = model.predict(tweet);

            boolean isPositive = pred >= 1.0f;

            if(isPositive){
                hit++;
            }
        }

        fIn = new FileInputStream(path + "test neg.txt");
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

    static void readData(String fileName, int label, boolean labelIntegrated) throws IOException {
        FileInputStream fIn = new FileInputStream(fileName);
        String[] data = new String(fIn.readAllBytes()).split("\3");

        int timer = 0;
        for(String tweet : data) {
            String text;
            int realLabel;

            if(!labelIntegrated) {
                text = tweet;
                realLabel = label;
            }
            else {
                String[] info = tweet.split("\2");

                text = info[0];
                realLabel = Integer.parseInt(info[1]);

                if(realLabel == 0) {
                    negativeCount++;
                }
                else {
                    positiveCount++;
                }
            }

            List<String> words = TextProcessing.lemmas(text);
            TextProcessing.removeStopWordsAndWeirdStrings(words, true, true, false);

            dataset.add(new Pair<>(words, realLabel));

            timer++;

            if(timer % 100 == 0) {
                System.out.println(timer + " / " + data.length + " scanned!");
            }
        }

        fIn.close();
        System.out.println("----------Training time!-------------\n\n");

        if(!labelIntegrated) {
            if(label == 0){
                negativeCount = data.length;
            }
            else{
                positiveCount = data.length;
            }
        }
    }
}