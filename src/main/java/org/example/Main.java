package org.example;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.simple.Sentence;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
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

    static Set<String> lemmas(String doc){
        Properties props = new Properties();
        props.put("annotators", "tokenize, ssplit, pos, lemma");

        StanfordCoreNLP pipeline;
        pipeline = new StanfordCoreNLP(props, false);
        Annotation document = pipeline.process(doc);

        Set<String> lemmas = new TreeSet<>();
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

    static void removeStopWordsAndWeirdStrings(Set<String> words) throws IOException {
        Set<String> removed = new TreeSet<>();

        for(String word : words){
            if(Pattern.compile("[!@#$:/.,]").matcher(word).find()){
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

        for(String shoudlRemoveString : removed){
            words.remove(shoudlRemoveString);
        }
    }

    static HashMap<String, FrequencyTable> establishTable(List<Pair<Set<String>, Integer>> dataset) {
        HashMap<String, FrequencyTable> table = new HashMap<>();

        for(Pair<Set<String>, Integer> pair : dataset){
            for(String word : pair.first) {
                if(!table.containsKey(word)){
                    table.put(word, new FrequencyTable(0, 0));
                }
                else{
                    FrequencyTable t = table.get(word);

                    if(pair.second == 0){
                        t.negativeOccurrence++;
                    }
                    else{
                        t.positiveOccurrence++;
                    }
                }
            }
        }

        return table;
    }
}

class NaiveBayes {

}

public class Main {
    public static void main(String[] args) throws IOException {

    }
}