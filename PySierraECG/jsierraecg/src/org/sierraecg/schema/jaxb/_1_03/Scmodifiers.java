//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.2.8-b130911.1802 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2014.08.15 at 08:44:37 PM EDT 
//


package org.sierraecg.schema.jaxb._1_03;

import java.math.BigInteger;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.XmlSchemaType;
import javax.xml.bind.annotation.XmlType;
import javax.xml.bind.annotation.XmlValue;


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
 *         &lt;element name="scmodifier" minOccurs="0">
 *           &lt;complexType>
 *             &lt;simpleContent>
 *               &lt;extension base="&lt;http://www.w3.org/2001/XMLSchema>string">
 *                 &lt;attribute name="scnumericcode" use="required" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" />
 *                 &lt;attribute name="scmodifiercode" use="required" type="{http://www.w3.org/2001/XMLSchema}string" />
 *                 &lt;attribute name="added" type="{http://www3.medical.philips.com}TYPEflag" />
 *                 &lt;attribute name="deleted" type="{http://www3.medical.philips.com}TYPEflag" />
 *               &lt;/extension>
 *             &lt;/simpleContent>
 *           &lt;/complexType>
 *         &lt;/element>
 *       &lt;/sequence>
 *       &lt;attribute name="changed" type="{http://www3.medical.philips.com}TYPEflag" default="False" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "", propOrder = {
    "scmodifier"
})
@XmlRootElement(name = "scmodifiers")
public class Scmodifiers {

    protected Scmodifiers.Scmodifier scmodifier;
    @XmlAttribute(name = "changed")
    protected TYPEflag changed;

    /**
     * Gets the value of the scmodifier property.
     * 
     * @return
     *     possible object is
     *     {@link Scmodifiers.Scmodifier }
     *     
     */
    public Scmodifiers.Scmodifier getScmodifier() {
        return scmodifier;
    }

    /**
     * Sets the value of the scmodifier property.
     * 
     * @param value
     *     allowed object is
     *     {@link Scmodifiers.Scmodifier }
     *     
     */
    public void setScmodifier(Scmodifiers.Scmodifier value) {
        this.scmodifier = value;
    }

    /**
     * Gets the value of the changed property.
     * 
     * @return
     *     possible object is
     *     {@link TYPEflag }
     *     
     */
    public TYPEflag getChanged() {
        if (changed == null) {
            return TYPEflag.FALSE;
        } else {
            return changed;
        }
    }

    /**
     * Sets the value of the changed property.
     * 
     * @param value
     *     allowed object is
     *     {@link TYPEflag }
     *     
     */
    public void setChanged(TYPEflag value) {
        this.changed = value;
    }


    /**
     * <p>Java class for anonymous complex type.
     * 
     * <p>The following schema fragment specifies the expected content contained within this class.
     * 
     * <pre>
     * &lt;complexType>
     *   &lt;simpleContent>
     *     &lt;extension base="&lt;http://www.w3.org/2001/XMLSchema>string">
     *       &lt;attribute name="scnumericcode" use="required" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" />
     *       &lt;attribute name="scmodifiercode" use="required" type="{http://www.w3.org/2001/XMLSchema}string" />
     *       &lt;attribute name="added" type="{http://www3.medical.philips.com}TYPEflag" />
     *       &lt;attribute name="deleted" type="{http://www3.medical.philips.com}TYPEflag" />
     *     &lt;/extension>
     *   &lt;/simpleContent>
     * &lt;/complexType>
     * </pre>
     * 
     * 
     */
    @XmlAccessorType(XmlAccessType.FIELD)
    @XmlType(name = "", propOrder = {
        "value"
    })
    public static class Scmodifier {

        @XmlValue
        protected String value;
        @XmlAttribute(name = "scnumericcode", required = true)
        @XmlSchemaType(name = "nonNegativeInteger")
        protected BigInteger scnumericcode;
        @XmlAttribute(name = "scmodifiercode", required = true)
        protected String scmodifiercode;
        @XmlAttribute(name = "added")
        protected TYPEflag added;
        @XmlAttribute(name = "deleted")
        protected TYPEflag deleted;

        /**
         * Gets the value of the value property.
         * 
         * @return
         *     possible object is
         *     {@link String }
         *     
         */
        public String getValue() {
            return value;
        }

        /**
         * Sets the value of the value property.
         * 
         * @param value
         *     allowed object is
         *     {@link String }
         *     
         */
        public void setValue(String value) {
            this.value = value;
        }

        /**
         * Gets the value of the scnumericcode property.
         * 
         * @return
         *     possible object is
         *     {@link BigInteger }
         *     
         */
        public BigInteger getScnumericcode() {
            return scnumericcode;
        }

        /**
         * Sets the value of the scnumericcode property.
         * 
         * @param value
         *     allowed object is
         *     {@link BigInteger }
         *     
         */
        public void setScnumericcode(BigInteger value) {
            this.scnumericcode = value;
        }

        /**
         * Gets the value of the scmodifiercode property.
         * 
         * @return
         *     possible object is
         *     {@link String }
         *     
         */
        public String getScmodifiercode() {
            return scmodifiercode;
        }

        /**
         * Sets the value of the scmodifiercode property.
         * 
         * @param value
         *     allowed object is
         *     {@link String }
         *     
         */
        public void setScmodifiercode(String value) {
            this.scmodifiercode = value;
        }

        /**
         * Gets the value of the added property.
         * 
         * @return
         *     possible object is
         *     {@link TYPEflag }
         *     
         */
        public TYPEflag getAdded() {
            return added;
        }

        /**
         * Sets the value of the added property.
         * 
         * @param value
         *     allowed object is
         *     {@link TYPEflag }
         *     
         */
        public void setAdded(TYPEflag value) {
            this.added = value;
        }

        /**
         * Gets the value of the deleted property.
         * 
         * @return
         *     possible object is
         *     {@link TYPEflag }
         *     
         */
        public TYPEflag getDeleted() {
            return deleted;
        }

        /**
         * Sets the value of the deleted property.
         * 
         * @param value
         *     allowed object is
         *     {@link TYPEflag }
         *     
         */
        public void setDeleted(TYPEflag value) {
            this.deleted = value;
        }

    }

}
