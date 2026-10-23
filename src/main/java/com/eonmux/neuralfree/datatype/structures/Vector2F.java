package com.eonmux.neuralfree.datatype.structures;

import java.util.ArrayList;

public class Vector2F {
    float F1, F2;

    public Vector2F(float f1, float f2) {
        F1 = f1;
        F2 = f2;
    }
    public float getF1() {
        return F1;
    }

    public void setF1(float f1) {
        F1 = f1;
    }

    public float getF2() {
        return F2;
    }

    public void setF2(float f2) {
        F2 = f2;
    }

    @Override
    public String toString() {
        StringBuilder serialized = new StringBuilder();
        serialized.append("[").append(F1).append(",").append(F2).append("]").append(System.lineSeparator());
        return serialized.toString();
    }

    public ArrayList<Float> asList() {
        ArrayList<Float> vectorAsLlist = new ArrayList<>();
        vectorAsLlist.add(this.F1);
        vectorAsLlist.add(this.F2);
        return vectorAsLlist;
    }
}
