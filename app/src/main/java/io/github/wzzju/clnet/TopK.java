package io.github.wzzju.clnet;

/**
 * Created by yuchen on 18-3-11.
 */

import java.util.*;

public class TopK<E extends Comparable<? super E>> {
    private PriorityQueue<E> queue;
    private int maxSize; //堆的最大容量

    public TopK(int maxSize) {
        if (maxSize <= 0) {
            throw new IllegalStateException();
        }
        this.maxSize = maxSize;
        this.queue = new PriorityQueue<>(maxSize, new Comparator<E>() {
            @Override
            public int compare(E left, E right) {
                // 最大堆用right - left，最小堆用left - right
                return (left.compareTo(right));
            }
        });
    }

    public void add(E e) {
        if (queue.size() < maxSize) {
            queue.add(e);
        } else {
            E peek = queue.peek();
            if (e.compareTo(peek) > 0) {
                queue.poll();
                queue.add(e);
            }
        }
    }

    public List<E> sortedList() {
        List<E> list = new ArrayList<>(queue);
        Collections.sort(list);
        return list;
    }

    public static void main(String[] args) {
        float[] result = new float[1000];
        for (int i = 999; i >= 0; i--) {
            result[i] = i * 1.0f;
        }
        TopK<Probability<Float>> topK = new TopK<>(5);
        for (int i = 0; i < result.length; i++) {
            topK.add(new Probability<>(result[i], i));
        }
        List<Probability<Float>> top = topK.sortedList();
        for (Probability<Float> e : top) {
            System.out.println("prob: " + e.getProb() + ", index: " + e.getIndex());
        }
    }
}