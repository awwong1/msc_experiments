//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.2.8-b130911.1802 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2014.08.15 at 08:44:37 PM EDT 
//


package org.sierraecg.schema.jaxb._1_03;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for TYPEnoiselevel.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="TYPEnoiselevel">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="None"/>
 *     &lt;enumeration value="Light"/>
 *     &lt;enumeration value="Marked"/>
 *     &lt;enumeration value="Severe"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlType(name = "TYPEnoiselevel")
@XmlEnum
public enum TYPEnoiselevel {

    @XmlEnumValue("None")
    NONE("None"),
    @XmlEnumValue("Light")
    LIGHT("Light"),
    @XmlEnumValue("Marked")
    MARKED("Marked"),
    @XmlEnumValue("Severe")
    SEVERE("Severe");
    private final String value;

    TYPEnoiselevel(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static TYPEnoiselevel fromValue(String v) {
        for (TYPEnoiselevel c: TYPEnoiselevel.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v);
    }

}
