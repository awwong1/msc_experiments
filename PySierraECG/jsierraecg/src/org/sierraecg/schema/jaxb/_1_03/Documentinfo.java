//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.2.8-b130911.1802 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2014.08.15 at 08:44:37 PM EDT 
//


package org.sierraecg.schema.jaxb._1_03;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.XmlSchemaType;
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
 *         &lt;element ref="{http://www3.medical.philips.com}documentname"/>
 *         &lt;element ref="{http://www3.medical.philips.com}documenttype"/>
 *         &lt;element ref="{http://www3.medical.philips.com}documentversion"/>
 *       &lt;/sequence>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "", propOrder = {
    "documentname",
    "documenttype",
    "documentversion"
})
@XmlRootElement(name = "documentinfo")
public class Documentinfo {

    @XmlElement(required = true)
    protected String documentname;
    @XmlElement(required = true)
    @XmlSchemaType(name = "string")
    protected TYPEschemaname documenttype;
    @XmlElement(required = true)
    protected String documentversion;

    /**
     * Gets the value of the documentname property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getDocumentname() {
        return documentname;
    }

    /**
     * Sets the value of the documentname property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setDocumentname(String value) {
        this.documentname = value;
    }

    /**
     * Gets the value of the documenttype property.
     * 
     * @return
     *     possible object is
     *     {@link TYPEschemaname }
     *     
     */
    public TYPEschemaname getDocumenttype() {
        return documenttype;
    }

    /**
     * Sets the value of the documenttype property.
     * 
     * @param value
     *     allowed object is
     *     {@link TYPEschemaname }
     *     
     */
    public void setDocumenttype(TYPEschemaname value) {
        this.documenttype = value;
    }

    /**
     * Gets the value of the documentversion property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getDocumentversion() {
        return documentversion;
    }

    /**
     * Sets the value of the documentversion property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setDocumentversion(String value) {
        this.documentversion = value;
    }

}
