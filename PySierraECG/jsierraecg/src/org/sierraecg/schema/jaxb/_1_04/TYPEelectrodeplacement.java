//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.2.8-b130911.1802 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2014.08.15 at 08:44:39 PM EDT 
//


package org.sierraecg.schema.jaxb._1_04;

import javax.xml.bind.annotation.XmlEnum;
import javax.xml.bind.annotation.XmlEnumValue;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for TYPEelectrodeplacement.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="TYPEelectrodeplacement">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="Unknown"/>
 *     &lt;enumeration value="STD"/>
 *     &lt;enumeration value="STD 12+"/>
 *     &lt;enumeration value="MASON-LIKAR"/>
 *     &lt;enumeration value="MASON-LIKAR 12+"/>
 *     &lt;enumeration value="MODIFIED"/>
 *     &lt;enumeration value="MODIFIED 12+"/>
 *     &lt;enumeration value="MIDA"/>
 *     &lt;enumeration value="EASI"/>
 *     &lt;enumeration value="EASI OFF STERNUM"/>
 *     &lt;enumeration value="FRANK"/>
 *     &lt;enumeration value="NEHB"/>
 *     &lt;enumeration value="Other"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlType(name = "TYPEelectrodeplacement")
@XmlEnum
public enum TYPEelectrodeplacement {

    @XmlEnumValue("Unknown")
    UNKNOWN("Unknown"),
    STD("STD"),
    @XmlEnumValue("STD 12+")
    STD_12("STD 12+"),
    @XmlEnumValue("MASON-LIKAR")
    MASON_LIKAR("MASON-LIKAR"),
    @XmlEnumValue("MASON-LIKAR 12+")
    MASON_LIKAR_12("MASON-LIKAR 12+"),
    MODIFIED("MODIFIED"),
    @XmlEnumValue("MODIFIED 12+")
    MODIFIED_12("MODIFIED 12+"),
    MIDA("MIDA"),
    EASI("EASI"),
    @XmlEnumValue("EASI OFF STERNUM")
    EASI_OFF_STERNUM("EASI OFF STERNUM"),
    FRANK("FRANK"),
    NEHB("NEHB"),
    @XmlEnumValue("Other")
    OTHER("Other");
    private final String value;

    TYPEelectrodeplacement(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static TYPEelectrodeplacement fromValue(String v) {
        for (TYPEelectrodeplacement c: TYPEelectrodeplacement.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v);
    }

}
