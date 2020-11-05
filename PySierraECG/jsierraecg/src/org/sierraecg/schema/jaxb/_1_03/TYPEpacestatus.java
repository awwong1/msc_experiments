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
 * <p>Java class for TYPEpacestatus.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * <p>
 * <pre>
 * &lt;simpleType name="TYPEpacestatus">
 *   &lt;restriction base="{http://www.w3.org/2001/XMLSchema}string">
 *     &lt;enumeration value="Paced"/>
 *     &lt;enumeration value="Non paced"/>
 *     &lt;enumeration value="Paced with magnet"/>
 *     &lt;enumeration value="Unknown"/>
 *   &lt;/restriction>
 * &lt;/simpleType>
 * </pre>
 * 
 */
@XmlType(name = "TYPEpacestatus")
@XmlEnum
public enum TYPEpacestatus {

    @XmlEnumValue("Paced")
    PACED("Paced"),
    @XmlEnumValue("Non paced")
    NON_PACED("Non paced"),
    @XmlEnumValue("Paced with magnet")
    PACED_WITH_MAGNET("Paced with magnet"),
    @XmlEnumValue("Unknown")
    UNKNOWN("Unknown");
    private final String value;

    TYPEpacestatus(String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    public static TYPEpacestatus fromValue(String v) {
        for (TYPEpacestatus c: TYPEpacestatus.values()) {
            if (c.value.equals(v)) {
                return c;
            }
        }
        throw new IllegalArgumentException(v);
    }

}
