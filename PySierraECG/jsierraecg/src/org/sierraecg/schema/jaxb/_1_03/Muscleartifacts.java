//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.2.8-b130911.1802 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2014.08.15 at 08:44:37 PM EDT 
//


package org.sierraecg.schema.jaxb._1_03;

import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for anonymous complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType>
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element ref="{http://www3.medical.philips.com}muscleartifact" maxOccurs="unbounded" minOccurs="0"/>
 *       &lt;/sequence>
 *       &lt;attribute name="level" use="required" type="{http://www3.medical.philips.com}TYPEnoiselevel" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "", propOrder = {
    "muscleartifact"
})
@XmlRootElement(name = "muscleartifacts")
public class Muscleartifacts {

    protected List<Muscleartifact> muscleartifact;
    @XmlAttribute(name = "level", required = true)
    protected TYPEnoiselevel level;

    /**
     * Gets the value of the muscleartifact property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the muscleartifact property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getMuscleartifact().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link Muscleartifact }
     * 
     * 
     */
    public List<Muscleartifact> getMuscleartifact() {
        if (muscleartifact == null) {
            muscleartifact = new ArrayList<Muscleartifact>();
        }
        return this.muscleartifact;
    }

    /**
     * Gets the value of the level property.
     * 
     * @return
     *     possible object is
     *     {@link TYPEnoiselevel }
     *     
     */
    public TYPEnoiselevel getLevel() {
        return level;
    }

    /**
     * Sets the value of the level property.
     * 
     * @param value
     *     allowed object is
     *     {@link TYPEnoiselevel }
     *     
     */
    public void setLevel(TYPEnoiselevel value) {
        this.level = value;
    }

}
