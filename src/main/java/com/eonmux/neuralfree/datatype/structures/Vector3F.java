package com.eonmux.neuralfree.datatype.structures;

import java.util.ArrayList;

public class Vector3F extends Vector2F {
    private Float F3;

    public Vector3F(float f1, float f2, float f3) {
        super(f1,f2);
        F3 = f3;
    }

    public Vector3F(Vector2F vector2F, Float f3) {
        super(vector2F.getF1(), vector2F.getF2());
        F3 = f3;
    }

    public float getF3() {
        return F3;
    }

    public void setF3(float f3) {
        F3 = f3;
    }

    @Override
    public String toString() {
        StringBuilder serialized = new StringBuilder();
        if(F3 != null) {
            serialized.append("[").append(F1).append(",").append(F2).append(",").append(F3).append("]").append(System.lineSeparator());
        } else {
            serialized.append("[").append(F1).append(",").append(F2).append("]").append(System.lineSeparator());
        }
        return serialized.toString();
    }

    @Override
    public ArrayList<Float> asList() {
        ArrayList<Float> vectorAsLlist = new ArrayList<>();
        vectorAsLlist.add(this.F1);
        vectorAsLlist.add(this.F2);
        vectorAsLlist.add(this.F3);
        return vectorAsLlist;
    }
}
