package io.github.wzzju.clnet;

public class Probability<T extends Comparable<? super T>> implements Comparable<Probability<T>> {
    private T prob;
    private int index;

    public Probability(T prob, int index) {
        this.prob = prob;
        this.index = index;
    }

    public T getProb() {
        return prob;
    }

    public int getIndex() {
        return index;
    }

    @Override
    public int compareTo(Probability<T> o) {
        return this.prob.compareTo(o.prob);
    }
}