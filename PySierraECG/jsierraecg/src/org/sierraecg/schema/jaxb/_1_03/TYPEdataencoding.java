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
 * <p>Java class for TYPEdataencoding.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="TYPEdataencoding">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="Plain"/>
 *     &lt;enumeration value="Base64"/>
 *     &lt;enumeration value="Hex"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlType(name = "TYPEdataencoding")
@XmlEnum
public enum TYPEdataencoding {

    @XmlEnumValue("Plain")
    PLAIN("Plain"),
    @XmlEnumValue("Base64")
    BASE_64("Base64"),
    @XmlEnumValue("Hex")
    HEX("Hex");
    private final String value;

    TYPEdataencoding(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static TYPEdataencoding fromValue(String v) {
        for (TYPEdataencoding c: TYPEdataencoding.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v);
    }

}
